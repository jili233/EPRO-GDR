#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "LLVMDemangle" for configuration "Release"
set_property(TARGET LLVMDemangle APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMDemangle PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMDemangle.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMDemangle.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMDemangle )
list(APPEND _cmake_import_check_files_for_LLVMDemangle "${_IMPORT_PREFIX}/lib/libLLVMDemangle.so.4.0.1" )

# Import target "LLVMSupport" for configuration "Release"
set_property(TARGET LLVMSupport APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMSupport PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMDemangle"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMSupport.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMSupport.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMSupport )
list(APPEND _cmake_import_check_files_for_LLVMSupport "${_IMPORT_PREFIX}/lib/libLLVMSupport.so.4.0.1" )

# Import target "LLVMTableGen" for configuration "Release"
set_property(TARGET LLVMTableGen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMTableGen PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMTableGen.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMTableGen.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMTableGen )
list(APPEND _cmake_import_check_files_for_LLVMTableGen "${_IMPORT_PREFIX}/lib/libLLVMTableGen.so.4.0.1" )

# Import target "llvm-tblgen" for configuration "Release"
set_property(TARGET llvm-tblgen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-tblgen PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-tblgen"
  )

list(APPEND _cmake_import_check_targets llvm-tblgen )
list(APPEND _cmake_import_check_files_for_llvm-tblgen "${_IMPORT_PREFIX}/bin/llvm-tblgen" )

# Import target "LLVMCore" for configuration "Release"
set_property(TARGET LLVMCore APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMCore PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMCore.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMCore.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMCore )
list(APPEND _cmake_import_check_files_for_LLVMCore "${_IMPORT_PREFIX}/lib/libLLVMCore.so.4.0.1" )

# Import target "LLVMIRReader" for configuration "Release"
set_property(TARGET LLVMIRReader APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMIRReader PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAsmParser;LLVMBitReader;LLVMCore;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMIRReader.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMIRReader.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMIRReader )
list(APPEND _cmake_import_check_files_for_LLVMIRReader "${_IMPORT_PREFIX}/lib/libLLVMIRReader.so.4.0.1" )

# Import target "LLVMCodeGen" for configuration "Release"
set_property(TARGET LLVMCodeGen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMCodeGen PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMBitReader;LLVMBitWriter;LLVMCore;LLVMMC;LLVMScalarOpts;LLVMSupport;LLVMTarget;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMCodeGen.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMCodeGen.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMCodeGen )
list(APPEND _cmake_import_check_files_for_LLVMCodeGen "${_IMPORT_PREFIX}/lib/libLLVMCodeGen.so.4.0.1" )

# Import target "LLVMSelectionDAG" for configuration "Release"
set_property(TARGET LLVMSelectionDAG APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMSelectionDAG PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMCodeGen;LLVMCore;LLVMMC;LLVMSupport;LLVMTarget;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMSelectionDAG.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMSelectionDAG.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMSelectionDAG )
list(APPEND _cmake_import_check_files_for_LLVMSelectionDAG "${_IMPORT_PREFIX}/lib/libLLVMSelectionDAG.so.4.0.1" )

# Import target "LLVMAsmPrinter" for configuration "Release"
set_property(TARGET LLVMAsmPrinter APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMAsmPrinter PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMCodeGen;LLVMCore;LLVMDebugInfoCodeView;LLVMDebugInfoMSF;LLVMMC;LLVMMCParser;LLVMSupport;LLVMTarget"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMAsmPrinter.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMAsmPrinter.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMAsmPrinter )
list(APPEND _cmake_import_check_files_for_LLVMAsmPrinter "${_IMPORT_PREFIX}/lib/libLLVMAsmPrinter.so.4.0.1" )

# Import target "LLVMMIRParser" for configuration "Release"
set_property(TARGET LLVMMIRParser APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMMIRParser PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAsmParser;LLVMCodeGen;LLVMCore;LLVMMC;LLVMSupport;LLVMTarget"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMMIRParser.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMMIRParser.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMMIRParser )
list(APPEND _cmake_import_check_files_for_LLVMMIRParser "${_IMPORT_PREFIX}/lib/libLLVMMIRParser.so.4.0.1" )

# Import target "LLVMGlobalISel" for configuration "Release"
set_property(TARGET LLVMGlobalISel APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMGlobalISel PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMCodeGen;LLVMCore;LLVMMC;LLVMSupport;LLVMTarget;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMGlobalISel.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMGlobalISel.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMGlobalISel )
list(APPEND _cmake_import_check_files_for_LLVMGlobalISel "${_IMPORT_PREFIX}/lib/libLLVMGlobalISel.so.4.0.1" )

# Import target "LLVMBitReader" for configuration "Release"
set_property(TARGET LLVMBitReader APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMBitReader PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCore;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMBitReader.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMBitReader.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMBitReader )
list(APPEND _cmake_import_check_files_for_LLVMBitReader "${_IMPORT_PREFIX}/lib/libLLVMBitReader.so.4.0.1" )

# Import target "LLVMBitWriter" for configuration "Release"
set_property(TARGET LLVMBitWriter APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMBitWriter PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMCore;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMBitWriter.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMBitWriter.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMBitWriter )
list(APPEND _cmake_import_check_files_for_LLVMBitWriter "${_IMPORT_PREFIX}/lib/libLLVMBitWriter.so.4.0.1" )

# Import target "LLVMTransformUtils" for configuration "Release"
set_property(TARGET LLVMTransformUtils APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMTransformUtils PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMCore;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMTransformUtils.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMTransformUtils.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMTransformUtils )
list(APPEND _cmake_import_check_files_for_LLVMTransformUtils "${_IMPORT_PREFIX}/lib/libLLVMTransformUtils.so.4.0.1" )

# Import target "LLVMInstrumentation" for configuration "Release"
set_property(TARGET LLVMInstrumentation APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMInstrumentation PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMCore;LLVMMC;LLVMProfileData;LLVMSupport;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMInstrumentation.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMInstrumentation.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMInstrumentation )
list(APPEND _cmake_import_check_files_for_LLVMInstrumentation "${_IMPORT_PREFIX}/lib/libLLVMInstrumentation.so.4.0.1" )

# Import target "LLVMInstCombine" for configuration "Release"
set_property(TARGET LLVMInstCombine APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMInstCombine PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMCore;LLVMSupport;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMInstCombine.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMInstCombine.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMInstCombine )
list(APPEND _cmake_import_check_files_for_LLVMInstCombine "${_IMPORT_PREFIX}/lib/libLLVMInstCombine.so.4.0.1" )

# Import target "LLVMScalarOpts" for configuration "Release"
set_property(TARGET LLVMScalarOpts APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMScalarOpts PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMCore;LLVMInstCombine;LLVMSupport;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMScalarOpts.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMScalarOpts.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMScalarOpts )
list(APPEND _cmake_import_check_files_for_LLVMScalarOpts "${_IMPORT_PREFIX}/lib/libLLVMScalarOpts.so.4.0.1" )

# Import target "LLVMipo" for configuration "Release"
set_property(TARGET LLVMipo APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMipo PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMBitWriter;LLVMCore;LLVMIRReader;LLVMInstCombine;LLVMInstrumentation;LLVMLinker;LLVMObject;LLVMProfileData;LLVMScalarOpts;LLVMSupport;LLVMTransformUtils;LLVMVectorize"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMipo.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMipo.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMipo )
list(APPEND _cmake_import_check_files_for_LLVMipo "${_IMPORT_PREFIX}/lib/libLLVMipo.so.4.0.1" )

# Import target "LLVMVectorize" for configuration "Release"
set_property(TARGET LLVMVectorize APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMVectorize PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMCore;LLVMSupport;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMVectorize.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMVectorize.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMVectorize )
list(APPEND _cmake_import_check_files_for_LLVMVectorize "${_IMPORT_PREFIX}/lib/libLLVMVectorize.so.4.0.1" )

# Import target "LLVMHello" for configuration "Release"
set_property(TARGET LLVMHello APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMHello PROPERTIES
  IMPORTED_COMMON_LANGUAGE_RUNTIME_RELEASE ""
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/LLVMHello.so"
  IMPORTED_NO_SONAME_RELEASE "TRUE"
  )

list(APPEND _cmake_import_check_targets LLVMHello )
list(APPEND _cmake_import_check_files_for_LLVMHello "${_IMPORT_PREFIX}/lib/LLVMHello.so" )

# Import target "LLVMObjCARCOpts" for configuration "Release"
set_property(TARGET LLVMObjCARCOpts APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMObjCARCOpts PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMCore;LLVMSupport;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMObjCARCOpts.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMObjCARCOpts.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMObjCARCOpts )
list(APPEND _cmake_import_check_files_for_LLVMObjCARCOpts "${_IMPORT_PREFIX}/lib/libLLVMObjCARCOpts.so.4.0.1" )

# Import target "LLVMCoroutines" for configuration "Release"
set_property(TARGET LLVMCoroutines APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMCoroutines PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMCore;LLVMScalarOpts;LLVMSupport;LLVMTransformUtils;LLVMipo"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMCoroutines.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMCoroutines.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMCoroutines )
list(APPEND _cmake_import_check_files_for_LLVMCoroutines "${_IMPORT_PREFIX}/lib/libLLVMCoroutines.so.4.0.1" )

# Import target "LLVMLinker" for configuration "Release"
set_property(TARGET LLVMLinker APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMLinker PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCore;LLVMSupport;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMLinker.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMLinker.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMLinker )
list(APPEND _cmake_import_check_files_for_LLVMLinker "${_IMPORT_PREFIX}/lib/libLLVMLinker.so.4.0.1" )

# Import target "LLVMAnalysis" for configuration "Release"
set_property(TARGET LLVMAnalysis APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMAnalysis PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCore;LLVMObject;LLVMProfileData;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMAnalysis.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMAnalysis.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMAnalysis )
list(APPEND _cmake_import_check_files_for_LLVMAnalysis "${_IMPORT_PREFIX}/lib/libLLVMAnalysis.so.4.0.1" )

# Import target "LLVMLTO" for configuration "Release"
set_property(TARGET LLVMLTO APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMLTO PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMBitReader;LLVMBitWriter;LLVMCodeGen;LLVMCore;LLVMInstCombine;LLVMLinker;LLVMMC;LLVMObjCARCOpts;LLVMObject;LLVMPasses;LLVMScalarOpts;LLVMSupport;LLVMTarget;LLVMTransformUtils;LLVMipo"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMLTO.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMLTO.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMLTO )
list(APPEND _cmake_import_check_files_for_LLVMLTO "${_IMPORT_PREFIX}/lib/libLLVMLTO.so.4.0.1" )

# Import target "LLVMMC" for configuration "Release"
set_property(TARGET LLVMMC APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMMC PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMMC.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMMC.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMMC )
list(APPEND _cmake_import_check_files_for_LLVMMC "${_IMPORT_PREFIX}/lib/libLLVMMC.so.4.0.1" )

# Import target "LLVMMCParser" for configuration "Release"
set_property(TARGET LLVMMCParser APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMMCParser PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMMCParser.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMMCParser.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMMCParser )
list(APPEND _cmake_import_check_files_for_LLVMMCParser "${_IMPORT_PREFIX}/lib/libLLVMMCParser.so.4.0.1" )

# Import target "LLVMMCDisassembler" for configuration "Release"
set_property(TARGET LLVMMCDisassembler APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMMCDisassembler PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMMCDisassembler.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMMCDisassembler.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMMCDisassembler )
list(APPEND _cmake_import_check_files_for_LLVMMCDisassembler "${_IMPORT_PREFIX}/lib/libLLVMMCDisassembler.so.4.0.1" )

# Import target "LLVMObject" for configuration "Release"
set_property(TARGET LLVMObject APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMObject PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMBitReader;LLVMCore;LLVMMC;LLVMMCParser;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMObject.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMObject.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMObject )
list(APPEND _cmake_import_check_files_for_LLVMObject "${_IMPORT_PREFIX}/lib/libLLVMObject.so.4.0.1" )

# Import target "LLVMObjectYAML" for configuration "Release"
set_property(TARGET LLVMObjectYAML APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMObjectYAML PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMObjectYAML.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMObjectYAML.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMObjectYAML )
list(APPEND _cmake_import_check_files_for_LLVMObjectYAML "${_IMPORT_PREFIX}/lib/libLLVMObjectYAML.so.4.0.1" )

# Import target "LLVMOption" for configuration "Release"
set_property(TARGET LLVMOption APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMOption PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMOption.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMOption.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMOption )
list(APPEND _cmake_import_check_files_for_LLVMOption "${_IMPORT_PREFIX}/lib/libLLVMOption.so.4.0.1" )

# Import target "LLVMDebugInfoDWARF" for configuration "Release"
set_property(TARGET LLVMDebugInfoDWARF APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMDebugInfoDWARF PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMObject;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMDebugInfoDWARF.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMDebugInfoDWARF.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMDebugInfoDWARF )
list(APPEND _cmake_import_check_files_for_LLVMDebugInfoDWARF "${_IMPORT_PREFIX}/lib/libLLVMDebugInfoDWARF.so.4.0.1" )

# Import target "LLVMDebugInfoMSF" for configuration "Release"
set_property(TARGET LLVMDebugInfoMSF APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMDebugInfoMSF PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMDebugInfoMSF.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMDebugInfoMSF.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMDebugInfoMSF )
list(APPEND _cmake_import_check_files_for_LLVMDebugInfoMSF "${_IMPORT_PREFIX}/lib/libLLVMDebugInfoMSF.so.4.0.1" )

# Import target "LLVMDebugInfoCodeView" for configuration "Release"
set_property(TARGET LLVMDebugInfoCodeView APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMDebugInfoCodeView PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMDebugInfoMSF;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMDebugInfoCodeView.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMDebugInfoCodeView.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMDebugInfoCodeView )
list(APPEND _cmake_import_check_files_for_LLVMDebugInfoCodeView "${_IMPORT_PREFIX}/lib/libLLVMDebugInfoCodeView.so.4.0.1" )

# Import target "LLVMDebugInfoPDB" for configuration "Release"
set_property(TARGET LLVMDebugInfoPDB APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMDebugInfoPDB PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMDebugInfoCodeView;LLVMDebugInfoMSF;LLVMObject;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMDebugInfoPDB.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMDebugInfoPDB.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMDebugInfoPDB )
list(APPEND _cmake_import_check_files_for_LLVMDebugInfoPDB "${_IMPORT_PREFIX}/lib/libLLVMDebugInfoPDB.so.4.0.1" )

# Import target "LLVMSymbolize" for configuration "Release"
set_property(TARGET LLVMSymbolize APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMSymbolize PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMDebugInfoDWARF;LLVMDebugInfoPDB;LLVMObject;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMSymbolize.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMSymbolize.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMSymbolize )
list(APPEND _cmake_import_check_files_for_LLVMSymbolize "${_IMPORT_PREFIX}/lib/libLLVMSymbolize.so.4.0.1" )

# Import target "LLVMExecutionEngine" for configuration "Release"
set_property(TARGET LLVMExecutionEngine APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMExecutionEngine PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCore;LLVMMC;LLVMObject;LLVMRuntimeDyld;LLVMSupport;LLVMTarget"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMExecutionEngine.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMExecutionEngine.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMExecutionEngine )
list(APPEND _cmake_import_check_files_for_LLVMExecutionEngine "${_IMPORT_PREFIX}/lib/libLLVMExecutionEngine.so.4.0.1" )

# Import target "LLVMInterpreter" for configuration "Release"
set_property(TARGET LLVMInterpreter APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMInterpreter PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCodeGen;LLVMCore;LLVMExecutionEngine;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMInterpreter.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMInterpreter.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMInterpreter )
list(APPEND _cmake_import_check_files_for_LLVMInterpreter "${_IMPORT_PREFIX}/lib/libLLVMInterpreter.so.4.0.1" )

# Import target "LLVMMCJIT" for configuration "Release"
set_property(TARGET LLVMMCJIT APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMMCJIT PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCore;LLVMExecutionEngine;LLVMObject;LLVMRuntimeDyld;LLVMSupport;LLVMTarget"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMMCJIT.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMMCJIT.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMMCJIT )
list(APPEND _cmake_import_check_files_for_LLVMMCJIT "${_IMPORT_PREFIX}/lib/libLLVMMCJIT.so.4.0.1" )

# Import target "LLVMOrcJIT" for configuration "Release"
set_property(TARGET LLVMOrcJIT APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMOrcJIT PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCore;LLVMExecutionEngine;LLVMObject;LLVMRuntimeDyld;LLVMSupport;LLVMTransformUtils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMOrcJIT.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMOrcJIT.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMOrcJIT )
list(APPEND _cmake_import_check_files_for_LLVMOrcJIT "${_IMPORT_PREFIX}/lib/libLLVMOrcJIT.so.4.0.1" )

# Import target "LLVMRuntimeDyld" for configuration "Release"
set_property(TARGET LLVMRuntimeDyld APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMRuntimeDyld PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMObject;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMRuntimeDyld.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMRuntimeDyld.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMRuntimeDyld )
list(APPEND _cmake_import_check_files_for_LLVMRuntimeDyld "${_IMPORT_PREFIX}/lib/libLLVMRuntimeDyld.so.4.0.1" )

# Import target "LLVMTarget" for configuration "Release"
set_property(TARGET LLVMTarget APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMTarget PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMCore;LLVMMC;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMTarget.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMTarget.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMTarget )
list(APPEND _cmake_import_check_files_for_LLVMTarget "${_IMPORT_PREFIX}/lib/libLLVMTarget.so.4.0.1" )

# Import target "LLVMX86CodeGen" for configuration "Release"
set_property(TARGET LLVMX86CodeGen APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMX86CodeGen PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMAsmPrinter;LLVMCodeGen;LLVMCore;LLVMGlobalISel;LLVMMC;LLVMSelectionDAG;LLVMSupport;LLVMTarget;LLVMX86AsmPrinter;LLVMX86Desc;LLVMX86Info;LLVMX86Utils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMX86CodeGen.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMX86CodeGen.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMX86CodeGen )
list(APPEND _cmake_import_check_files_for_LLVMX86CodeGen "${_IMPORT_PREFIX}/lib/libLLVMX86CodeGen.so.4.0.1" )

# Import target "LLVMX86AsmParser" for configuration "Release"
set_property(TARGET LLVMX86AsmParser APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMX86AsmParser PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMMCParser;LLVMSupport;LLVMX86Desc;LLVMX86Info"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMX86AsmParser.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMX86AsmParser.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMX86AsmParser )
list(APPEND _cmake_import_check_files_for_LLVMX86AsmParser "${_IMPORT_PREFIX}/lib/libLLVMX86AsmParser.so.4.0.1" )

# Import target "LLVMX86Disassembler" for configuration "Release"
set_property(TARGET LLVMX86Disassembler APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMX86Disassembler PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMCDisassembler;LLVMSupport;LLVMX86Info"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMX86Disassembler.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMX86Disassembler.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMX86Disassembler )
list(APPEND _cmake_import_check_files_for_LLVMX86Disassembler "${_IMPORT_PREFIX}/lib/libLLVMX86Disassembler.so.4.0.1" )

# Import target "LLVMX86AsmPrinter" for configuration "Release"
set_property(TARGET LLVMX86AsmPrinter APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMX86AsmPrinter PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMSupport;LLVMX86Utils"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMX86AsmPrinter.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMX86AsmPrinter.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMX86AsmPrinter )
list(APPEND _cmake_import_check_files_for_LLVMX86AsmPrinter "${_IMPORT_PREFIX}/lib/libLLVMX86AsmPrinter.so.4.0.1" )

# Import target "LLVMX86Desc" for configuration "Release"
set_property(TARGET LLVMX86Desc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMX86Desc PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMMC;LLVMMCDisassembler;LLVMObject;LLVMSupport;LLVMX86AsmPrinter;LLVMX86Info"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMX86Desc.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMX86Desc.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMX86Desc )
list(APPEND _cmake_import_check_files_for_LLVMX86Desc "${_IMPORT_PREFIX}/lib/libLLVMX86Desc.so.4.0.1" )

# Import target "LLVMX86Info" for configuration "Release"
set_property(TARGET LLVMX86Info APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMX86Info PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMX86Info.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMX86Info.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMX86Info )
list(APPEND _cmake_import_check_files_for_LLVMX86Info "${_IMPORT_PREFIX}/lib/libLLVMX86Info.so.4.0.1" )

# Import target "LLVMX86Utils" for configuration "Release"
set_property(TARGET LLVMX86Utils APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMX86Utils PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCore;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMX86Utils.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMX86Utils.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMX86Utils )
list(APPEND _cmake_import_check_files_for_LLVMX86Utils "${_IMPORT_PREFIX}/lib/libLLVMX86Utils.so.4.0.1" )

# Import target "LLVMAsmParser" for configuration "Release"
set_property(TARGET LLVMAsmParser APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMAsmParser PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCore;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMAsmParser.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMAsmParser.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMAsmParser )
list(APPEND _cmake_import_check_files_for_LLVMAsmParser "${_IMPORT_PREFIX}/lib/libLLVMAsmParser.so.4.0.1" )

# Import target "LLVMLineEditor" for configuration "Release"
set_property(TARGET LLVMLineEditor APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMLineEditor PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMLineEditor.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMLineEditor.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMLineEditor )
list(APPEND _cmake_import_check_files_for_LLVMLineEditor "${_IMPORT_PREFIX}/lib/libLLVMLineEditor.so.4.0.1" )

# Import target "LLVMProfileData" for configuration "Release"
set_property(TARGET LLVMProfileData APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMProfileData PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCore;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMProfileData.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMProfileData.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMProfileData )
list(APPEND _cmake_import_check_files_for_LLVMProfileData "${_IMPORT_PREFIX}/lib/libLLVMProfileData.so.4.0.1" )

# Import target "LLVMCoverage" for configuration "Release"
set_property(TARGET LLVMCoverage APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMCoverage PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMCore;LLVMObject;LLVMProfileData;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMCoverage.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMCoverage.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMCoverage )
list(APPEND _cmake_import_check_files_for_LLVMCoverage "${_IMPORT_PREFIX}/lib/libLLVMCoverage.so.4.0.1" )

# Import target "LLVMPasses" for configuration "Release"
set_property(TARGET LLVMPasses APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMPasses PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMAnalysis;LLVMCodeGen;LLVMCore;LLVMInstCombine;LLVMInstrumentation;LLVMScalarOpts;LLVMSupport;LLVMTransformUtils;LLVMVectorize;LLVMipo"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMPasses.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMPasses.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMPasses )
list(APPEND _cmake_import_check_files_for_LLVMPasses "${_IMPORT_PREFIX}/lib/libLLVMPasses.so.4.0.1" )

# Import target "LLVMLibDriver" for configuration "Release"
set_property(TARGET LLVMLibDriver APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMLibDriver PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMObject;LLVMOption;LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMLibDriver.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMLibDriver.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMLibDriver )
list(APPEND _cmake_import_check_files_for_LLVMLibDriver "${_IMPORT_PREFIX}/lib/libLLVMLibDriver.so.4.0.1" )

# Import target "LLVMXRay" for configuration "Release"
set_property(TARGET LLVMXRay APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LLVMXRay PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMSupport"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLLVMXRay.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLLVMXRay.so.4"
  )

list(APPEND _cmake_import_check_targets LLVMXRay )
list(APPEND _cmake_import_check_files_for_LLVMXRay "${_IMPORT_PREFIX}/lib/libLLVMXRay.so.4.0.1" )

# Import target "LTO" for configuration "Release"
set_property(TARGET LTO APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(LTO PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "LLVMX86CodeGen;LLVMX86AsmPrinter;LLVMX86AsmParser;LLVMX86Desc;LLVMX86Info;LLVMX86Disassembler;LLVMBitReader;LLVMCore;LLVMLTO;LLVMMC;LLVMMCDisassembler;LLVMSupport;LLVMTarget"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libLTO.so.4.0.1"
  IMPORTED_SONAME_RELEASE "libLTO.so.4"
  )

list(APPEND _cmake_import_check_targets LTO )
list(APPEND _cmake_import_check_files_for_LTO "${_IMPORT_PREFIX}/lib/libLTO.so.4.0.1" )

# Import target "llvm-ar" for configuration "Release"
set_property(TARGET llvm-ar APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-ar PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-ar"
  )

list(APPEND _cmake_import_check_targets llvm-ar )
list(APPEND _cmake_import_check_files_for_llvm-ar "${_IMPORT_PREFIX}/bin/llvm-ar" )

# Import target "llvm-config" for configuration "Release"
set_property(TARGET llvm-config APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-config PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-config"
  )

list(APPEND _cmake_import_check_targets llvm-config )
list(APPEND _cmake_import_check_files_for_llvm-config "${_IMPORT_PREFIX}/bin/llvm-config" )

# Import target "llvm-lto" for configuration "Release"
set_property(TARGET llvm-lto APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-lto PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-lto"
  )

list(APPEND _cmake_import_check_targets llvm-lto )
list(APPEND _cmake_import_check_files_for_llvm-lto "${_IMPORT_PREFIX}/bin/llvm-lto" )

# Import target "llvm-profdata" for configuration "Release"
set_property(TARGET llvm-profdata APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-profdata PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-profdata"
  )

list(APPEND _cmake_import_check_targets llvm-profdata )
list(APPEND _cmake_import_check_files_for_llvm-profdata "${_IMPORT_PREFIX}/bin/llvm-profdata" )

# Import target "bugpoint" for configuration "Release"
set_property(TARGET bugpoint APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(bugpoint PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/bugpoint"
  )

list(APPEND _cmake_import_check_targets bugpoint )
list(APPEND _cmake_import_check_files_for_bugpoint "${_IMPORT_PREFIX}/bin/bugpoint" )

# Import target "BugpointPasses" for configuration "Release"
set_property(TARGET BugpointPasses APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(BugpointPasses PROPERTIES
  IMPORTED_COMMON_LANGUAGE_RUNTIME_RELEASE ""
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/BugpointPasses.so"
  IMPORTED_NO_SONAME_RELEASE "TRUE"
  )

list(APPEND _cmake_import_check_targets BugpointPasses )
list(APPEND _cmake_import_check_files_for_BugpointPasses "${_IMPORT_PREFIX}/lib/BugpointPasses.so" )

# Import target "llvm-dsymutil" for configuration "Release"
set_property(TARGET llvm-dsymutil APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-dsymutil PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-dsymutil"
  )

list(APPEND _cmake_import_check_targets llvm-dsymutil )
list(APPEND _cmake_import_check_files_for_llvm-dsymutil "${_IMPORT_PREFIX}/bin/llvm-dsymutil" )

# Import target "llc" for configuration "Release"
set_property(TARGET llc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llc PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llc"
  )

list(APPEND _cmake_import_check_targets llc )
list(APPEND _cmake_import_check_files_for_llc "${_IMPORT_PREFIX}/bin/llc" )

# Import target "lli" for configuration "Release"
set_property(TARGET lli APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(lli PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/lli"
  )

list(APPEND _cmake_import_check_targets lli )
list(APPEND _cmake_import_check_files_for_lli "${_IMPORT_PREFIX}/bin/lli" )

# Import target "llvm-as" for configuration "Release"
set_property(TARGET llvm-as APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-as PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-as"
  )

list(APPEND _cmake_import_check_targets llvm-as )
list(APPEND _cmake_import_check_files_for_llvm-as "${_IMPORT_PREFIX}/bin/llvm-as" )

# Import target "llvm-bcanalyzer" for configuration "Release"
set_property(TARGET llvm-bcanalyzer APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-bcanalyzer PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-bcanalyzer"
  )

list(APPEND _cmake_import_check_targets llvm-bcanalyzer )
list(APPEND _cmake_import_check_files_for_llvm-bcanalyzer "${_IMPORT_PREFIX}/bin/llvm-bcanalyzer" )

# Import target "llvm-c-test" for configuration "Release"
set_property(TARGET llvm-c-test APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-c-test PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-c-test"
  )

list(APPEND _cmake_import_check_targets llvm-c-test )
list(APPEND _cmake_import_check_files_for_llvm-c-test "${_IMPORT_PREFIX}/bin/llvm-c-test" )

# Import target "llvm-cat" for configuration "Release"
set_property(TARGET llvm-cat APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-cat PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-cat"
  )

list(APPEND _cmake_import_check_targets llvm-cat )
list(APPEND _cmake_import_check_files_for_llvm-cat "${_IMPORT_PREFIX}/bin/llvm-cat" )

# Import target "llvm-cov" for configuration "Release"
set_property(TARGET llvm-cov APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-cov PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-cov"
  )

list(APPEND _cmake_import_check_targets llvm-cov )
list(APPEND _cmake_import_check_files_for_llvm-cov "${_IMPORT_PREFIX}/bin/llvm-cov" )

# Import target "llvm-cxxdump" for configuration "Release"
set_property(TARGET llvm-cxxdump APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-cxxdump PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-cxxdump"
  )

list(APPEND _cmake_import_check_targets llvm-cxxdump )
list(APPEND _cmake_import_check_files_for_llvm-cxxdump "${_IMPORT_PREFIX}/bin/llvm-cxxdump" )

# Import target "llvm-cxxfilt" for configuration "Release"
set_property(TARGET llvm-cxxfilt APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-cxxfilt PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-cxxfilt"
  )

list(APPEND _cmake_import_check_targets llvm-cxxfilt )
list(APPEND _cmake_import_check_files_for_llvm-cxxfilt "${_IMPORT_PREFIX}/bin/llvm-cxxfilt" )

# Import target "llvm-diff" for configuration "Release"
set_property(TARGET llvm-diff APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-diff PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-diff"
  )

list(APPEND _cmake_import_check_targets llvm-diff )
list(APPEND _cmake_import_check_files_for_llvm-diff "${_IMPORT_PREFIX}/bin/llvm-diff" )

# Import target "llvm-dis" for configuration "Release"
set_property(TARGET llvm-dis APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-dis PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-dis"
  )

list(APPEND _cmake_import_check_targets llvm-dis )
list(APPEND _cmake_import_check_files_for_llvm-dis "${_IMPORT_PREFIX}/bin/llvm-dis" )

# Import target "llvm-dwarfdump" for configuration "Release"
set_property(TARGET llvm-dwarfdump APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-dwarfdump PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-dwarfdump"
  )

list(APPEND _cmake_import_check_targets llvm-dwarfdump )
list(APPEND _cmake_import_check_files_for_llvm-dwarfdump "${_IMPORT_PREFIX}/bin/llvm-dwarfdump" )

# Import target "llvm-dwp" for configuration "Release"
set_property(TARGET llvm-dwp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-dwp PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-dwp"
  )

list(APPEND _cmake_import_check_targets llvm-dwp )
list(APPEND _cmake_import_check_files_for_llvm-dwp "${_IMPORT_PREFIX}/bin/llvm-dwp" )

# Import target "llvm-extract" for configuration "Release"
set_property(TARGET llvm-extract APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-extract PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-extract"
  )

list(APPEND _cmake_import_check_targets llvm-extract )
list(APPEND _cmake_import_check_files_for_llvm-extract "${_IMPORT_PREFIX}/bin/llvm-extract" )

# Import target "llvm-link" for configuration "Release"
set_property(TARGET llvm-link APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-link PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-link"
  )

list(APPEND _cmake_import_check_targets llvm-link )
list(APPEND _cmake_import_check_files_for_llvm-link "${_IMPORT_PREFIX}/bin/llvm-link" )

# Import target "llvm-lto2" for configuration "Release"
set_property(TARGET llvm-lto2 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-lto2 PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-lto2"
  )

list(APPEND _cmake_import_check_targets llvm-lto2 )
list(APPEND _cmake_import_check_files_for_llvm-lto2 "${_IMPORT_PREFIX}/bin/llvm-lto2" )

# Import target "llvm-mc" for configuration "Release"
set_property(TARGET llvm-mc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-mc PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-mc"
  )

list(APPEND _cmake_import_check_targets llvm-mc )
list(APPEND _cmake_import_check_files_for_llvm-mc "${_IMPORT_PREFIX}/bin/llvm-mc" )

# Import target "llvm-mcmarkup" for configuration "Release"
set_property(TARGET llvm-mcmarkup APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-mcmarkup PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-mcmarkup"
  )

list(APPEND _cmake_import_check_targets llvm-mcmarkup )
list(APPEND _cmake_import_check_files_for_llvm-mcmarkup "${_IMPORT_PREFIX}/bin/llvm-mcmarkup" )

# Import target "llvm-modextract" for configuration "Release"
set_property(TARGET llvm-modextract APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-modextract PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-modextract"
  )

list(APPEND _cmake_import_check_targets llvm-modextract )
list(APPEND _cmake_import_check_files_for_llvm-modextract "${_IMPORT_PREFIX}/bin/llvm-modextract" )

# Import target "llvm-nm" for configuration "Release"
set_property(TARGET llvm-nm APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-nm PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-nm"
  )

list(APPEND _cmake_import_check_targets llvm-nm )
list(APPEND _cmake_import_check_files_for_llvm-nm "${_IMPORT_PREFIX}/bin/llvm-nm" )

# Import target "llvm-objdump" for configuration "Release"
set_property(TARGET llvm-objdump APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-objdump PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-objdump"
  )

list(APPEND _cmake_import_check_targets llvm-objdump )
list(APPEND _cmake_import_check_files_for_llvm-objdump "${_IMPORT_PREFIX}/bin/llvm-objdump" )

# Import target "llvm-opt-report" for configuration "Release"
set_property(TARGET llvm-opt-report APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-opt-report PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-opt-report"
  )

list(APPEND _cmake_import_check_targets llvm-opt-report )
list(APPEND _cmake_import_check_files_for_llvm-opt-report "${_IMPORT_PREFIX}/bin/llvm-opt-report" )

# Import target "llvm-pdbdump" for configuration "Release"
set_property(TARGET llvm-pdbdump APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-pdbdump PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-pdbdump"
  )

list(APPEND _cmake_import_check_targets llvm-pdbdump )
list(APPEND _cmake_import_check_files_for_llvm-pdbdump "${_IMPORT_PREFIX}/bin/llvm-pdbdump" )

# Import target "llvm-readobj" for configuration "Release"
set_property(TARGET llvm-readobj APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-readobj PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-readobj"
  )

list(APPEND _cmake_import_check_targets llvm-readobj )
list(APPEND _cmake_import_check_files_for_llvm-readobj "${_IMPORT_PREFIX}/bin/llvm-readobj" )

# Import target "llvm-rtdyld" for configuration "Release"
set_property(TARGET llvm-rtdyld APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-rtdyld PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-rtdyld"
  )

list(APPEND _cmake_import_check_targets llvm-rtdyld )
list(APPEND _cmake_import_check_files_for_llvm-rtdyld "${_IMPORT_PREFIX}/bin/llvm-rtdyld" )

# Import target "llvm-size" for configuration "Release"
set_property(TARGET llvm-size APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-size PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-size"
  )

list(APPEND _cmake_import_check_targets llvm-size )
list(APPEND _cmake_import_check_files_for_llvm-size "${_IMPORT_PREFIX}/bin/llvm-size" )

# Import target "llvm-split" for configuration "Release"
set_property(TARGET llvm-split APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-split PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-split"
  )

list(APPEND _cmake_import_check_targets llvm-split )
list(APPEND _cmake_import_check_files_for_llvm-split "${_IMPORT_PREFIX}/bin/llvm-split" )

# Import target "llvm-stress" for configuration "Release"
set_property(TARGET llvm-stress APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-stress PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-stress"
  )

list(APPEND _cmake_import_check_targets llvm-stress )
list(APPEND _cmake_import_check_files_for_llvm-stress "${_IMPORT_PREFIX}/bin/llvm-stress" )

# Import target "llvm-strings" for configuration "Release"
set_property(TARGET llvm-strings APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-strings PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-strings"
  )

list(APPEND _cmake_import_check_targets llvm-strings )
list(APPEND _cmake_import_check_files_for_llvm-strings "${_IMPORT_PREFIX}/bin/llvm-strings" )

# Import target "llvm-symbolizer" for configuration "Release"
set_property(TARGET llvm-symbolizer APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-symbolizer PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-symbolizer"
  )

list(APPEND _cmake_import_check_targets llvm-symbolizer )
list(APPEND _cmake_import_check_files_for_llvm-symbolizer "${_IMPORT_PREFIX}/bin/llvm-symbolizer" )

# Import target "llvm-xray" for configuration "Release"
set_property(TARGET llvm-xray APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(llvm-xray PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/llvm-xray"
  )

list(APPEND _cmake_import_check_targets llvm-xray )
list(APPEND _cmake_import_check_files_for_llvm-xray "${_IMPORT_PREFIX}/bin/llvm-xray" )

# Import target "obj2yaml" for configuration "Release"
set_property(TARGET obj2yaml APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(obj2yaml PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/obj2yaml"
  )

list(APPEND _cmake_import_check_targets obj2yaml )
list(APPEND _cmake_import_check_files_for_obj2yaml "${_IMPORT_PREFIX}/bin/obj2yaml" )

# Import target "opt" for configuration "Release"
set_property(TARGET opt APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(opt PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/opt"
  )

list(APPEND _cmake_import_check_targets opt )
list(APPEND _cmake_import_check_files_for_opt "${_IMPORT_PREFIX}/bin/opt" )

# Import target "sancov" for configuration "Release"
set_property(TARGET sancov APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(sancov PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/sancov"
  )

list(APPEND _cmake_import_check_targets sancov )
list(APPEND _cmake_import_check_files_for_sancov "${_IMPORT_PREFIX}/bin/sancov" )

# Import target "sanstats" for configuration "Release"
set_property(TARGET sanstats APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(sanstats PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/sanstats"
  )

list(APPEND _cmake_import_check_targets sanstats )
list(APPEND _cmake_import_check_files_for_sanstats "${_IMPORT_PREFIX}/bin/sanstats" )

# Import target "verify-uselistorder" for configuration "Release"
set_property(TARGET verify-uselistorder APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(verify-uselistorder PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/verify-uselistorder"
  )

list(APPEND _cmake_import_check_targets verify-uselistorder )
list(APPEND _cmake_import_check_files_for_verify-uselistorder "${_IMPORT_PREFIX}/bin/verify-uselistorder" )

# Import target "yaml2obj" for configuration "Release"
set_property(TARGET yaml2obj APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(yaml2obj PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/yaml2obj"
  )

list(APPEND _cmake_import_check_targets yaml2obj )
list(APPEND _cmake_import_check_files_for_yaml2obj "${_IMPORT_PREFIX}/bin/yaml2obj" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
