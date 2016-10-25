# Try to find a bash completion target to install our bash completion script
# Once done it will define
# - BASHCOMPLETION_FOUND
# - BASHCOMPLETION_PATH

# We won't be doing anything fancy just check if the folder 'bash_completion.d'
# exists

if(IS_DIRECTORY "/etc/bash_completion.d/")
    set(BASHCOMPLETION_PATH "/etc/bash_completion.d")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    BashCompletion
    DEFAULT_MSG
    BASHCOMPLETION_PATH
)
