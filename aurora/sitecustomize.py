"""Aurora startup compatibility for the released JAX oneAPI plugin.

The 0.11.1 development wheel enables command buffers in its compile defaults,
while Aurora's Level Zero runtime does not implement them.  This XLA option is
an enum/list (not a Boolean): an empty value disables every command type.
Inject that compiler override into every JAX compilation before any user code
runs.

This file is added to ``PYTHONPATH`` only by ``runtime_env.sh``. Import errors
are intentionally ignored so bootstrap can create a fresh environment before
JAX and the oneAPI plugin have been installed.
"""

try:
    from jax._src import compiler as _compiler
except ImportError:
    pass
else:
    _original_get_compile_options = _compiler.get_compile_options

    def _get_compile_options_without_command_buffers(*args, **kwargs):
        argument_list = list(args)
        if len(argument_list) > 3:
            overrides = argument_list[3]
            argument_list[3] = {**(overrides or {}),
                                'xla_gpu_enable_command_buffer': ''}
        else:
            overrides = kwargs.get('env_options_overrides')
            kwargs['env_options_overrides'] = {
                **(overrides or {}), 'xla_gpu_enable_command_buffer': ''}
        return _original_get_compile_options(*argument_list, **kwargs)

    _compiler.get_compile_options = _get_compile_options_without_command_buffers
