import tvm.micro
import tvm.micro.testing
# To check whether microTVM is enabled or not
print(tvm.micro.testing.get_supported_boards("c"))
