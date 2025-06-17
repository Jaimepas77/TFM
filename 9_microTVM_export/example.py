import os
import numpy as np
import pathlib
import json
from PIL import Image
import tarfile

import tvm
from tvm import relay
from tvm.relay.backend import Executor, Runtime
from tvm.contrib.download import download_testdata
from tvm.micro import export_model_library_format
from tvm.relay.op.contrib import cmsisnn
from tvm.micro.testing.utils import create_header_file

MODEL_URL = "https://github.com/mlcommons/tiny/raw/bceb91c5ad2e2deb295547d81505721d3a87d578/benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite"
MODEL_NAME = "vww_96_int8.tflite"
MODEL_PATH = download_testdata(MODEL_URL, MODEL_NAME, module="model")

tflite_model_buf = open(MODEL_PATH, "rb").read()
import tflite

tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)

input_shape = (1, 96, 96, 3)
INPUT_NAME = "input_1_int8"

relay_mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict={INPUT_NAME: input_shape}, dtype_dict={INPUT_NAME: "int8"}
)

# We can use TVM native schedules or rely on the CMSIS-NN kernels using TVM Bring-Your-Own-Code (BYOC) capability.
USE_CMSIS_NN = True

# USMP (Unified Static Memory Planning) performs memory planning of all tensors holistically to achieve best memory utilization
DISABLE_USMP = False

# USe the crt
RUNTIME = Runtime("crt")

# We define the target by passing the board name to `tvm.target.target.micro`.
# If your board is not included in the supported models, you can define the target such as:
TARGET = tvm.target.Target("c -keys=arm_cpu,cpu -mcpu=cortex-m7")
# TARGET = tvm.target.target.micro("stm3214r5zi")

# Use AOT executor, unpacked API and C calling style.
EXECUTOR = tvm.relay.backend.Executor(
    "aot", {"unpacked-api": True, "interface-api": "c", "workspace-byte-alignment": 8}
)

# Now, we set the compilation configurations and compile the model for the target:
config = {"tir.disable_vectorize": True}
if USE_CMSIS_NN:
    config["relay.ext.cmsisnn.options"] = {"mcpu": TARGET.mcpu}
if DISABLE_USMP:
    config["tir.usmp.enable"] = False

with tvm.transform.PassContext(opt_level=3, config=config):
    if USE_CMSIS_NN:
        # When we are using CMSIS-NN, TVM searches for patterns in the
        # relay graph that it can offload to the CMSIS-NN kernels.
        relay_mod = cmsisnn.partition_for_cmsisnn(relay_mod, params, mcpu=TARGET.mcpu)
    lowered = tvm.relay.build(
        relay_mod, target=TARGET, params=params, runtime=RUNTIME, executor=EXECUTOR
    )
parameter_size = len(tvm.runtime.save_param_dict(lowered.get_params()))
print(f"Model parameter size: {parameter_size}")

# We need to pick a directory where our file will be saved.
# If running on Google Colab, we'll save everything in ``/root/tutorial`` (aka ``~/tutorial``)
# but you'll probably want to store it elsewhere if running locally.

# BUILD_DIR = pathlib.Path("/root/tutorial")
BUILD_DIR = pathlib.Path("/home/jaime/Documentos/TFM/microTVM/9_microTVM_export/build")

BUILD_DIR.mkdir(exist_ok=True)

# Export into tar file
TAR_PATH = pathlib.Path(BUILD_DIR) / "model.tar"
export_model_library_format(lowered, TAR_PATH)

with tarfile.open(TAR_PATH, mode="a") as tar_file:
    SAMPLES_DIR = "samples"
    SAMPLE_PERSON_URL = (
        "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/data/vww_sample_person.jpg"
    )
    SAMPLE_NOT_PERSON_URL = "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/data/vww_sample_not_person.jpg"

    SAMPLE_PERSON_PATH = download_testdata(SAMPLE_PERSON_URL, "person.jpg", module=SAMPLES_DIR)
    img = Image.open(SAMPLE_PERSON_PATH)
    create_header_file("sample_person", np.asanyarray(img), SAMPLES_DIR, tar_file)

    SAMPLE_NOT_PERSON_PATH = download_testdata(
        SAMPLE_NOT_PERSON_URL, "not_person.jpg", module=SAMPLES_DIR
    )
    img = Image.open(SAMPLE_NOT_PERSON_PATH)
    create_header_file("sample_not_person", np.asanyarray(img), SAMPLES_DIR, tar_file)


