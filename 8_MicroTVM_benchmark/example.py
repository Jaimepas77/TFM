# Variables que se pueden exportar
# TVM_MICRO_BOARD (string) - ek_ra8m1
# 
import os
import pathlib
import tarfile
import shutil

import tensorflow as tf
import numpy as np

import tvm
from tvm import relay
from tvm.relay.backend import Executor, Runtime
from tvm.contrib.download import download_testdata
from tvm.micro import export_model_library_format
import tvm.micro.testing
from tvm.micro.testing.utils import (
    create_header_file,
    mlf_extract_workspace_size_bytes,
)

# Seleccionar el modelo a usar
MODEL_INDEX = 4 # 1: KWS, 2: VWW, 4: IC

# Parámetros del modelo
MODEL_SHORT_NAME = "ERROR"
MODEL_URL = ""
MODEL_FILE_NAME = ""

if MODEL_INDEX == 1:
    MODEL_SHORT_NAME = "KWS"
    MODEL_URL = "https://github.com/mlcommons/tiny/raw/bceb91c5ad2e2deb295547d81505721d3a87d578/benchmark/training/keyword_spotting/trained_models/kws_ref_model.tflite"
    MODEL_FILE_NAME = "kws_ref_model.tflite"
elif MODEL_INDEX == 2:
    MODEL_SHORT_NAME = "VWW"
    MODEL_URL = "https://github.com/mlcommons/tiny/raw/bceb91c5ad2e2deb295547d81505721d3a87d578/benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite"
    MODEL_FILE_NAME = "vww_96_int8.tflite"
elif MODEL_INDEX == 4:
    MODEL_SHORT_NAME = "IC"
    MODEL_URL = "https://github.com/mlcommons/tiny/raw/bceb91c5ad2e2deb295547d81505721d3a87d578/benchmark/training/image_classification/trained_models/pretrainedResnet_quant.tflite"
    MODEL_FILE_NAME = "pretrainedResnet_quant.tflite"

MODEL_PATH = download_testdata(MODEL_URL, MODEL_FILE_NAME, module="model", overwrite=True)

tflite_model_buf = open(MODEL_PATH, "rb").read()

import tflite

tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)

interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_name = input_details[0]["name"]
input_shape = tuple(input_details[0]["shape"])
input_dtype = np.dtype(input_details[0]["dtype"]).name
output_name =output_details[0]["name"]
output_shape = tuple(output_details[0]["shape"])
output_dtype = np.dtype(output_details[0]["dtype"]).name

# Extraemos la información de cuantización del modelo TFLite.
# Esto es necesario porque enviamos datos cuantizados al intérprete
# desde el host
quant_output_scale = output_details[0]["quantization_parameters"]["scales"][0]
quant_output_zero_point = output_details[0]["quantization_parameters"]["zero_points"][0]

relay_mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict={input_name: input_shape}, dtype_dict={input_name: input_dtype}
)

# Usar el C runtime (crt)
RUNTIME = Runtime("crt") # Caso AoT
# RUNTIME = Runtime("crt", {"system-lib": True}) # Caso Graph

# Usar el ejecutor AoT con `unpacked-api=True` y `interface-api=c`. `interface-api=c` fuerza
# al compilador a generar APIs de función de tipo C y `unpacked-api=True` fuerza al compilador
# a generar entradas de formato desempaquetado mínimo, lo que reduce el uso de memoria de pila al llamar
# a las capas de inferencia del modelo.
EXECUTOR = Executor(
    "aot", {"unpacked-api": True, "interface-api": "c", "workspace-byte-alignment": 8},
    # "graph", {"link-params": True}, # Creo que graph podría estropear el benchmarking, no usar
)

# Seleccionar una placa Zephyr (export TVM_MICRO_BOARD = tuplacafavorita) 
BOARD = os.getenv("TVM_MICRO_BOARD", default="nucleo_h743zi") # ek_ra8m1

# Obtener la descripción completa del objetivo usando BOARD
TARGET = tvm.micro.testing.get_target("zephyr", BOARD)
# TARGET = tvm.target.Target("c -keys=arm_cpu,cpu -mcpu=cortex-m85")
# TARGET = tvm.target.Target("llvm -keys=arm_cpu,cpu -mcpu=cortex-m85")

config = {"tir.disable_vectorize": True} 
with tvm.transform.PassContext(opt_level=3, config=config):
    module = tvm.relay.build(
        relay_mod, target=TARGET, params=params, runtime=RUNTIME, executor=EXECUTOR
    )

temp_dir = tvm.contrib.utils.tempdir()
model_tar_path = temp_dir / "model.tar"
export_model_library_format(module, model_tar_path)
workspace_size = mlf_extract_workspace_size_bytes(model_tar_path)

extra_tar_dir = tvm.contrib.utils.tempdir()
extra_tar_file = extra_tar_dir / "extra.tar"

with tarfile.open(extra_tar_file, "w:gz") as tf:
    create_header_file(
        "output_data",
        np.zeros(
            shape=output_shape,
            dtype=output_dtype,
        ),
        "include/tvm",
        tf,
    )

input_total_size = 1
for i in range(len(input_shape)):
    input_total_size *= input_shape[i]

template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("zephyr"))
project_options = {
    "extra_files_tar": str(extra_tar_file),
    "project_type": "mlperftiny",
    "board": BOARD,
    "compile_definitions": [
        f"-DWORKSPACE_SIZE={workspace_size + 512}",
        f"-DTARGET_MODEL={MODEL_INDEX}", # Índice del modelo para la compilación
        f"-DTH_MODEL_VERSION=EE_MODEL_VERSION_{MODEL_SHORT_NAME}01", # Como lo requiere la API MLPerfTiny
        f"-DMAX_DB_INPUT_SIZE={input_total_size}", # Tamaño máximo del array de datos de entrada
    ],
}

project_options["compile_definitions"].append(f"-DOUT_QUANT_SCALE={quant_output_scale}")
project_options["compile_definitions"].append(f"-DOUT_QUANT_ZERO={quant_output_zero_point}")

# Nota: ajustar según la placa de destino
project_options["config_main_stack_size"] = 4000

generated_project_dir = temp_dir / "project"

project = tvm.micro.project.generate_project_from_mlf(
    template_project_path, generated_project_dir, model_tar_path, project_options
)

print(f"Proyecto generado en {generated_project_dir}")
# input("Presiona Enter para continuar...")
project.build()

if(BOARD == "nucleo_h743zi"):
    with open(f'{generated_project_dir}/build/CMakeCache.txt', 'a') as file:
        file.write('ZEPHYR_BOARD_FLASH_RUNNER:STRING=openocd\n')
elif(BOARD == "ek_ra8m1"):
    with open(f'{generated_project_dir}/build/CMakeCache.txt', 'a') as file:
        file.write('ZEPHYR_BOARD_FLASH_RUNNER:STRING=jlink\n')

#Limpiar el directorio BUILD y cosas extra
shutil.rmtree(generated_project_dir / "build")
(generated_project_dir / "model.tar").unlink()

project_tar_path = pathlib.Path(os.getcwd()) / "project.tar"
with tarfile.open(project_tar_path, "w:tar") as tar:
    tar.add(generated_project_dir, arcname=os.path.basename("project"))

print(f"El proyecto generado se encuentra aquí: {project_tar_path}")
