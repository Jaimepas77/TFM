import os

use_physical_hw = bool(os.getenv("TVM_MICRO_USE_HW"))

import json
import numpy as np
import pathlib

import tvm
from tvm.relay.backend import Runtime
import tvm.micro.testing

# Define model
data_shape = (1, 3, 10, 10)
weight_shape = (6, 3, 5, 5)

data = tvm.relay.var("data", tvm.relay.TensorType(data_shape, "float32"))
weight = tvm.relay.var("weight", tvm.relay.TensorType(weight_shape, "float32"))

y = tvm.relay.nn.conv2d(
    data,
    weight,
    padding=(2, 2),
    kernel_size=(5, 5),
    kernel_layout="OIHW",
    out_dtype="float32",
)
f = tvm.relay.Function([data, weight], y)

relay_mod = tvm.IRModule.from_expr(f)
relay_mod = tvm.relay.transform.InferType()(relay_mod)

weight_sample = np.random.rand(
    weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]
).astype("float32")
params = {"weight": weight_sample}

# Define target (micro device)
RUNTIME = Runtime("crt", {"system-lib": True})
TARGET = tvm.micro.testing.get_target("crt")

# Compiling for physical hardware
# --------------------------------------------------------------------------
#  When running on physical hardware, choose a TARGET and a BOARD that describe the hardware. The
#  STM32L4R5ZI Nucleo target and board is chosen in the example below.
if use_physical_hw:
    BOARD = os.getenv("TVM_MICRO_BOARD", default="nucleo_h743zi")
    SERIAL = os.getenv("TVM_MICRO_SERIAL", default=None)
    TARGET = tvm.micro.testing.get_target("zephyr", BOARD)

# Extract tunning tasks (picking the real tasks)
pass_context = tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True})
with pass_context:
    tasks = tvm.autotvm.task.extract_from_program(relay_mod["main"], {}, TARGET)
assert len(tasks) > 0

# Configure microTVM for autotunning and build
module_loader = tvm.micro.AutoTvmModuleLoader(
    template_project_dir=pathlib.Path(tvm.micro.get_microtvm_template_projects("crt")),
    project_options={"verbose": False},
)
builder = tvm.autotvm.LocalBuilder(
    n_parallel=1,
    build_kwargs={"build_option": {"tir.disable_vectorize": True}},
    do_fork=True,
    build_func=tvm.micro.autotvm_build_func,
    runtime=RUNTIME,
)
runner = tvm.autotvm.LocalRunner(number=1, repeat=1, timeout=100, module_loader=module_loader)

measure_option = tvm.autotvm.measure_option(builder=builder, runner=runner)

# Compiling for physical hardware
if use_physical_hw:
    module_loader = tvm.micro.AutoTvmModuleLoader(
        template_project_dir=pathlib.Path(tvm.micro.get_microtvm_template_projects("zephyr")),
        project_options={
            "board": BOARD,
            "verbose": False,
            "project_type": "host_driven",
            "serial_number": SERIAL,
        },
    )
    builder = tvm.autotvm.LocalBuilder(
        n_parallel=1,
        build_kwargs={"build_option": {"tir.disable_vectorize": True}},
        do_fork=False,
        build_func=tvm.micro.autotvm_build_func,
        runtime=RUNTIME,
    )
    runner = tvm.autotvm.LocalRunner(number=1, repeat=1, timeout=100, module_loader=module_loader)

    measure_option = tvm.autotvm.measure_option(builder=builder, runner=runner)

# Run autotunning
autotune_log_file = pathlib.Path("microtvm_autotune.log.txt")
if os.path.exists(autotune_log_file):
    os.remove(autotune_log_file)

num_trials = 10
for task in tasks:
    tuner = tvm.autotvm.tuner.GATuner(task)
    tuner.tune(
        n_trial=num_trials,
        measure_option=measure_option,
        callbacks=[
            tvm.autotvm.callback.log_to_file(str(autotune_log_file)),
            tvm.autotvm.callback.progress_bar(num_trials, si_prefix="M"),
        ],
        si_prefix="M",
    )


####
# Build and Run without autotunning
with pass_context:
    lowered = tvm.relay.build(relay_mod, target=TARGET, runtime=RUNTIME, params=params)

temp_dir = tvm.contrib.utils.tempdir()
project = tvm.micro.generate_project(
    str(tvm.micro.get_microtvm_template_projects("crt")),
    lowered,
    temp_dir / "project",
    {"verbose": False},
)

# Compiling for physical hardware
if use_physical_hw:
    temp_dir = tvm.contrib.utils.tempdir()
    project = tvm.micro.generate_project(
        str(tvm.micro.get_microtvm_template_projects("zephyr")),
        lowered,
        temp_dir / "project",
        {
            "board": BOARD,
            "verbose": False,
            "project_type": "host_driven",
            "serial_number": SERIAL,
            "config_main_stack_size": 4096,
        },
    )

project.build()

generated_project_dir = temp_dir / "project"
if(BOARD == "nucleo_h743zi"):
    with open(f'{generated_project_dir}/build/CMakeCache.txt', 'a') as file:
        file.write('ZEPHYR_BOARD_FLASH_RUNNER:STRING=openocd\n')
elif(BOARD == "ek_ra8m1"):
    with open(f'{generated_project_dir}/build/CMakeCache.txt', 'a') as file:
        file.write('ZEPHYR_BOARD_FLASH_RUNNER:STRING=jlink\n')

project.flash()
with tvm.micro.Session(project.transport()) as session:
    debug_module = tvm.micro.create_local_debug_executor(
        lowered.get_graph_json(), session.get_system_lib(), session.device
    )
    debug_module.set_input(**lowered.get_params())
    print("########## Build without Autotuning ##########")
    debug_module.run()
    del debug_module
###

###
# Build and run with autotunning
with tvm.autotvm.apply_history_best(str(autotune_log_file)):
    with pass_context:
        lowered_tuned = tvm.relay.build(relay_mod, target=TARGET, runtime=RUNTIME, params=params)

temp_dir = tvm.contrib.utils.tempdir()
project = tvm.micro.generate_project(
    str(tvm.micro.get_microtvm_template_projects("crt")),
    lowered_tuned,
    temp_dir / "project",
    {"verbose": False},
)

# Compiling for physical hardware
if use_physical_hw:
    temp_dir = tvm.contrib.utils.tempdir()
    project = tvm.micro.generate_project(
        str(tvm.micro.get_microtvm_template_projects("zephyr")),
        lowered_tuned,
        temp_dir / "project",
        {
            "board": BOARD,
            "verbose": False,
            "project_type": "host_driven",
            "serial_number": SERIAL,
            "config_main_stack_size": 4096,
        },
    )

project.build()

generated_project_dir = temp_dir / "project"
if(BOARD == "nucleo_h743zi"):
    with open(f'{generated_project_dir}/build/CMakeCache.txt', 'a') as file:
        file.write('ZEPHYR_BOARD_FLASH_RUNNER:STRING=openocd\n')
elif(BOARD == "ek_ra8m1"):
    with open(f'{generated_project_dir}/build/CMakeCache.txt', 'a') as file:
        file.write('ZEPHYR_BOARD_FLASH_RUNNER:STRING=jlink\n')

project.flash()
with tvm.micro.Session(project.transport()) as session:
    debug_module = tvm.micro.create_local_debug_executor(
        lowered_tuned.get_graph_json(), session.get_system_lib(), session.device
    )
    debug_module.set_input(**lowered_tuned.get_params())
    print("########## Build with Autotuning ##########")
    debug_module.run()
    del debug_module
###