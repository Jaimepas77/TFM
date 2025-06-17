import os

# By default, this tutorial runs on x86 CPU using TVM's C runtime.
# If you would like to run on real Zephyr hardware, you must
# export the `TVM_MICRO_USE_HW` environment variable.
use_physical_hw = True# bool(os.getenv("TVM_MICRO_USE_HW"))

import json
import tarfile
import pathlib
import tempfile
import numpy as np

import tvm
import tvm.micro
import tvm.micro.testing
from tvm import relay
import tvm.contrib.utils
from tvm.micro import export_model_library_format
from tvm.contrib.download import download_testdata

# Import model
model_url = (
    "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/model/sine_model.tflite"
)

model_file = "sine_model.tflite"
model_path = download_testdata(model_url, model_file, module="data")

tflite_model_buf = open(model_path, "rb").read()

try:
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model

    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

version = tflite_model.Version()
print("Model Version: " + str(version))

input_tensor = "dense_4_input"
input_shape = (1,)
input_dtype = "float32"

mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict={input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype}
)

# Define target
input_tensor = "dense_4_input"
input_shape = (1,)
input_dtype = "float32"

mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict={input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype}
)

RUNTIME = tvm.relay.backend.Runtime("crt", {"system-lib": True})
TARGET = tvm.micro.testing.get_target("crt")

# When running on physical hardware, choose a TARGET and a BOARD that describe the hardware. The
# STM32L4R5ZI Nucleo target and board is chosen in the example below. You could change the testing
# board by simply exporting `TVM_MICRO_BOARD` variable with a different Zephyr supported board.

if use_physical_hw:
    BOARD = os.getenv("TVM_MICRO_BOARD", default="qemu_x86")
    SERIAL = os.getenv("TVM_MICRO_SERIAL", default=None)
    print(tvm.micro.testing.utils.get_supported_boards("zephyr"))
    TARGET = tvm.micro.testing.get_target("zephyr", BOARD)

# Compile for the target
with tvm.transform.PassContext(opt_level=2, config={"tir.disable_vectorize": True}):
    module = relay.build(mod, target=TARGET, runtime=RUNTIME, params=params)

# Inspect compilation results (C code)
c_source_module = module.get_lib().imported_modules[0]
assert c_source_module.type_key == "c", "tutorial is broken"

c_source_code = c_source_module.get_source()
first_few_lines = c_source_code.split("\n")[:10]
assert any(
    l.startswith("TVM_DLL int32_t tvmgen_default_") for l in first_few_lines
), f"tutorial is broken: {first_few_lines!r}"
print("\n".join(first_few_lines))

# After generating the C implementation, we have to integrate it into
# a project. To do this we can use MLF (Model Library Format)

# Get a temporary path where we can store the tarball (since this is running as a tutorial).

temp_dir = tvm.contrib.utils.tempdir()
model_tar_path = temp_dir / "model.tar"
export_model_library_format(module, model_tar_path)

with tarfile.open(model_tar_path, "r:*") as tar_f:
    print("\n".join(f" - {m.name}" for m in tar_f.getmembers()))

# TVM also provides a standard way for embedded platforms to automatically generate a standalone
# project, compile and flash it to a target, and communicate with it using the standard TVM RPC
# protocol. The Model Library Format serves as the model input to this process. When embedded
# platforms provide such an integration, they can be used directly by TVM for both host-driven
# inference and autotuning . This integration is provided by the
# `microTVM Project API` <https://github.com/apache/tvm-rfcs/blob/main/rfcs/0008-microtvm-project-api.md>_,
#
# Embedded platforms need to provide a Template Project containing a microTVM API Server (typically,
# this lives in a file ``microtvm_api_server.py`` in the root directory). Let's use the example ``host``
# project in this tutorial, which simulates the device using a POSIX subprocess and pipes:

template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("crt"))
project_options = {}  # You can use options to provide platform-specific options through TVM.

#  For physical hardware, you can try out the Zephyr platform by using a different template project
#  and options:

if use_physical_hw:
    template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("zephyr"))
    project_options = {
        "project_type": "host_driven",
        "board": BOARD,
        "serial_number": SERIAL,
        "config_main_stack_size": 4096,
        "zephyr_base": os.getenv("ZEPHYR_BASE", default="/home/jaime/zephyrproject/zephyr"),
    }

# Create a temporary directory
# temp_dir = tvm.contrib.utils.tempdir()
generated_project_dir = temp_dir / "generated-project"
generated_project = tvm.micro.generate_project(
    template_project_path, module, generated_project_dir, project_options
)

# Build and flash the project
generated_project.build()
if(BOARD == "nucleo_h743zi"):
    with open(f'{generated_project_dir}/build/CMakeCache.txt', 'a') as file:
        file.write('ZEPHYR_BOARD_FLASH_RUNNER:STRING=openocd\n')
elif(BOARD == "ek_ra8m1"):
    with open(f'{generated_project_dir}/build/CMakeCache.txt', 'a') as file:
        file.write('ZEPHYR_BOARD_FLASH_RUNNER:STRING=jlink\n')

generated_project.flash()

with tvm.micro.Session(transport_context_manager=generated_project.transport()) as session:
    graph_mod = tvm.micro.create_local_graph_executor(
        module.get_graph_json(), session.get_system_lib(), session.device
    )

    # Set the model parameters using the lowered parameters produced by `relay.build`.
    graph_mod.set_input(**module.get_params())

    # The model consumes a single float32 value and returns a predicted sine value.  To pass the
    # input value we construct a tvm.nd.array object with a single contrived number as input. For
    # this model values of 0 to 2Pi are acceptable.
    graph_mod.set_input(input_tensor, tvm.nd.array(np.array([0.5], dtype="float32")))
    graph_mod.run()

    tvm_output = graph_mod.get_output(0).numpy()
    print("result is: " + str(tvm_output))

