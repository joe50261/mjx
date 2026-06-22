set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(Threads REQUIRED)
# Ubuntu's libprotobuf-dev does not ship a CONFIG package (ProtobufConfig.cmake),
# only the module-mode FindProtobuf.cmake. Prefer CONFIG, fall back to MODULE.
find_package(Protobuf CONFIG QUIET)
if(NOT Protobuf_FOUND)
  find_package(Protobuf MODULE REQUIRED)
endif()
find_package(gRPC CONFIG REQUIRED)

message(STATUS "Using Protobuf ${Protobuf_VERSION}")
message(STATUS "Using gRPC ${gRPC_VERSION}")

set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
find_program(_PROTOBUF_PROTOC protoc)
set(_GRPC_GRPCPP gRPC::grpc++)
set(_REFLECTION gRPC::grpc++_reflection)
find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
