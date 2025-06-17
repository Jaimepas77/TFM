import os

# By default, this tutorial runs on x86 CPU using TVM's C runtime. If you would like
# to run on real Zephyr hardware, you must export the `TVM_MICRO_USE_HW` environment
# variable. Otherwise (if you are using the C runtime), you can skip installing
# Zephyr. It takes ~20 minutes to install Zephyr.
use_physical_hw = bool(os.getenv("TVM_MICRO_USE_HW"))
# use_physical_hw = True

# --Import model
import numpy as np
import pathlib
import json

import tvm
from tvm import relay
import tvm.micro.testing
from tvm.relay.backend import Executor, Runtime
from tvm.contrib.download import download_testdata

MODEL_URL = "https://github.com/mlcommons/tiny/raw/bceb91c5ad2e2deb295547d81505721d3a87d578/benchmark/training/keyword_spotting/trained_models/kws_ref_model.tflite"
MODEL_PATH = download_testdata(MODEL_URL, "kws_ref_model.tflite", module="model")
SAMPLE_URL = "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/data/keyword_spotting_int8_6.pyc.npy"
SAMPLE_PATH = download_testdata(SAMPLE_URL, "keyword_spotting_int8_6.pyc.npy", module="data")

tflite_model_buf = open(MODEL_PATH, "rb").read()
try:
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model

    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

input_shape = (1, 49, 10, 1)
INPUT_NAME = "input_1"
relay_mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict={INPUT_NAME: input_shape}, dtype_dict={INPUT_NAME: "int8"}
)

# --Define target
# Use the C runtime (crt) and enable static linking by setting system-lib to True
RUNTIME = Runtime("crt", {"system-lib": True})

# Simulate a microcontroller on the host machine. Uses the main() from `src/runtime/crt/host/main.cc`.
# To use physical hardware, replace "host" with something matching your hardware.
TARGET = tvm.micro.testing.get_target("crt")

# Use the AOT executor rather than graph or vm executors. Don't use unpacked API or C calling style.
EXECUTOR = Executor("aot")

if use_physical_hw:
    # Export TVM_MICRO_BOARD with the name of the physical hw in case of use
    BOARD = os.getenv("TVM_MICRO_BOARD", default="ek_ra8m1")
    SERIAL = os.getenv("TVM_MICRO_SERIAL", default=None)
    TARGET = tvm.micro.testing.get_target("zephyr", BOARD)

# --Compile
with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
    module = tvm.relay.build(
        relay_mod, target=TARGET, params=params, runtime=RUNTIME, executor=EXECUTOR
    )

# --Create microTVM project
template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("crt"))
project_options = {}  # You can use options to provide platform-specific options through TVM.

if use_physical_hw:
    template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("zephyr"))
    project_options = {
        "project_type": "host_driven",
        "board": BOARD,
        "serial_number": SERIAL,
        "config_main_stack_size": 4096,
        "zephyr_base": os.getenv("ZEPHYR_BASE", default="/content/zephyrproject/zephyr"),
    }

temp_dir = tvm.contrib.utils.tempdir()
generated_project_dir = temp_dir / "project"
project = tvm.micro.generate_project(
    template_project_path, module, generated_project_dir, project_options
)

# --Build, flash and execute
project.build()
if(BOARD == "nucleo_h743zi"):
    with open(f'{generated_project_dir}/build/CMakeCache.txt', 'a') as file:
        file.write('ZEPHYR_BOARD_FLASH_RUNNER:STRING=openocd\n')
elif(BOARD == "ek_ra8m1"):
    with open(f'{generated_project_dir}/build/CMakeCache.txt', 'a') as file:
        file.write('ZEPHYR_BOARD_FLASH_RUNNER:STRING=jlink\n')

project.flash()

labels = [
    "_silence_",
    "_unknown_",
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
]
with tvm.micro.Session(project.transport()) as session:
    aot_executor = tvm.runtime.executor.aot_executor.AotModule(session.create_aot_executor())
    sample = np.load(SAMPLE_PATH)
    aot_executor.get_input(INPUT_NAME).copyfrom(sample)
    aot_executor.run()
    result = aot_executor.get_output(0).numpy()
    print(f"Label is `{labels[np.argmax(result)]}` with index `{np.argmax(result)}`")


