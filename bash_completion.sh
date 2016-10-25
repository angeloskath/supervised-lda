# Completions for the programs
lda_commands="transform train"
lda_train=$(echo "--help" "--quiet" "--workers" "--topics" "--iterations"  \
    "--random_state" "--snapshot_every" "--continue" "--e_step_iterations" \
    "--e_step_tolerance" "--compute_likelihood" "--initialize_seeded"      \
    "--initialize_random")
lda_transform=$(echo "--help" "--quiet" "--workers" "--e_step_iterations"  \
    "--e_step_tolerance")

slda_commands="transform train"
slda_train=$(echo "--help" "--quiet" "--workers" "--topics" "--iterations"  \
    "--random_state" "--snapshot_every" "--continue" "--e_step_iterations"  \
    "--e_step_tolerance" "--compute_likelihood" "--fixed_point_iteration"   \
    "--m_step_iterations" "--m_step_tolerance" "--regularization_penalty"   \
    "--initialize_seeded" "--initialize_random")
slda_transform=$(echo "--help" "--quiet" "--workers" "--e_step_iterations"  \
    "--e_step_tolerance")

fslda_commands="transform train online_train"
fslda_train=$(echo "--help" "--quiet" "--workers" "--topics" "--iterations"   \
    "--random_state" "--snapshot_every" "--continue" "--e_step_iterations"    \
    "--e_step_tolerance" "--compute_likelihood"                               \
    "--m_step_iterations" "--m_step_tolerance" "--continue_from_unsupervised" \
    "--supervised_weight" "--regularization_penalty" "--initialize_seeded"    \
    "--initialize_random")
fslda_online_train=$(echo "--help" "--quiet" "--workers" "--topics"   \
    "--iterations" "--random_state" "--snapshot_every" "--continue"   \
    "--e_step_iterations" "--e_step_tolerance" "--compute_likelihood" \
    "--batch_size" "--momentum" "--learning_rate" "--beta_weight"     \
    "--continue_from_unsupervised" "--supervised_weight"              \
    "--regularization_penalty" "--initialize_seeded" "--initialize_random")
fslda_transform=$(echo "--help" "--quiet" "--workers" "--e_step_iterations"  \
    "--e_step_tolerance")

_ldaplusplus()
{
    local cur   # Current partial option
    local prog  # The program name
    local cmd   # The command (train, transform, ...)
    local var   # A variable to be used for bash indirection

    cur=${COMP_WORDS[COMP_CWORD]}
    prog=${COMP_WORDS[0]}

    if (( ${#COMP_WORDS[*]} == 2 )); then
        # We need a command so choose from commands
        var="${prog}_commands"
        COMPREPLY=($(compgen -W "${!var}" -- $cur))
    else
        # We need an option
        cmd=${COMP_WORDS[1]}
        var="${prog}_${cmd}"

        case $cur in
            -*)
                COMPREPLY=($(compgen -W "${!var}" -- $cur))
                ;;
            *)
                COMPREPLY=($(compgen -f -- $cur))
                ;;
        esac
    fi

    return 0
}

complete -F _ldaplusplus lda
complete -F _ldaplusplus slda
complete -F _ldaplusplus fslda
