@echo off

set CMD_LINE_ARGS=
:setupArgs
if ""%1""=="""" goto run
set CMD_LINE_ARGS=%CMD_LINE_ARGS% %1
shift
goto setupArgs

:run
java -cp mulan.jar;weka.jar;commons-cli-1.2.jar; mulan.console.CommandDispatcher %CMD_LINE_ARGS%
