import os
import ruamel.yaml as yaml
import shutil


from azureml.core import Experiment, RunConfiguration, Workspace
from azureml.pipeline.core import PipelineData, Pipeline
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.conda_dependencies import CondaDependencies


class DefaultConstants:
    WORKSPACE_NAME = 'zhizhu-test-ws'
    SUBSCRIPTION_ID = 'e9b2ec51-5c94-4fa8-809a-dc1e695e4896'
    RESOURCE_GROUP = 'zhizhu-test-ws-rg'
    COMPUTE_NAME = 'zhizhu-compute'


def get_default_workspace():
    return Workspace.get(
        name=DefaultConstants.WORKSPACE_NAME,
        subscription_id=DefaultConstants.SUBSCRIPTION_ID,
        resource_group=DefaultConstants.RESOURCE_GROUP
    )


class Port:
    def __init__(self, name, type_, description, options=[]):
        self.name = name
        self.type = type_
        self.description = description
        self.options = options
        self.value = None  # to be set by code.
        self._connected = False
        self._assigned = False

    @staticmethod
    def from_dct(dct: dict):
        options = []
        if 'options' in dct:
            for option in dct.get('options'):
                if isinstance(option, dict):
                    for name, child_port_dicts in option.items():
                        for child_port_dict in child_port_dicts:
                            print(child_port_dict.get('name'))
                            options.append(Port.from_dct(child_port_dict))

        return Port(
            name=dct.get('name'),
            type_=dct.get('type'),
            description=dct.get('description'),
            options=options
        )

    def prepare(self):
        def _regular_name(port_name):
            # AML Service does not allow name with spaces. Replace them with underscore.
            return '_'.join(port_name.split())

        if not self.prepared:
            conn = PipelineData(_regular_name(self.name))
            self.value = conn

    @property
    def prepared(self):
        return self.value is not None

    @property
    def connected(self):
        return self._connected

    def connect(self, another_port):
        if not another_port:
            raise ValueError(f"Cannot connect to a empty port")
        if not another_port.prepared:
            raise ValueError(f"Port({another_port.name}) is not ready yet")

        self.value = another_port.value
        self._connected = True

    def assign(self, value):
        self.value = value
        self._assigned = True


class Module:
    def __init__(self, spec_file_path, source_directory):
        self._file = spec_file_path
        self._src_dir = source_directory

        with open(self._file) as f:
            self._dct = yaml.safe_load(f)

        self._input_ports = [Port.from_dct(p) for p in self._get_value('inputs')]
        self._output_ports = [Port.from_dct(p) for p in self._get_value('outputs')]

        for p in self._output_ports:
            p.prepare()

    def _get_value(self, key_path):
        if not key_path:
            raise ValueError("key_path must not be empty")
        if not self._dct:
            raise ValueError("dct is empty")

        segments = key_path.split('/')

        walked = []

        cur_obj = self._dct
        for seg in segments:
            if cur_obj is None:
                raise ValueError(f"Missing {'/'.join(walked)} block in yaml file: {self._file}")
            if not isinstance(cur_obj, dict):
                raise ValueError(f"Block {'/'.join(walked)} cannot contain a child. "
                                 f"Confirm yaml file: {self._file}")

            cur_obj = cur_obj.get(seg)
            walked.append(seg)

        if cur_obj is None:
            raise ValueError(f"Missing {'/'.join(walked)} block in yaml file: {self._file}")
        return cur_obj

    def assign_parameter(self, port_name, value):
        self.params[port_name].assign(value)
        for child in self.params[port_name].options:
            self._input_ports.append(child)

    @property
    def name(self):
        return self._get_value('name')

    @property
    def description(self):
        return self._get_value('description')

    @property
    def inputs(self):
        return {p.name: p for p in self._input_ports}

    @property
    def input_refs(self):
        return [p.value for p in self._input_ports if p.connected]

    @property
    def params(self):
        disconnected_inputs = [p for p in self._input_ports if not p.connected]
        return {p.name: p for p in disconnected_inputs}

    @property
    def outputs(self):
        return {p.name: p for p in self._output_ports}

    @property
    def output_refs(self):
        return [p.value for p in self._output_ports]

    @property
    def image(self):
        try:
            return self._get_value('implementation/container/image')
        except Exception as e:
            print(str(e))
            return None

    @property
    def conda_dependencies(self):
        return CondaDependencies(_underlying_structure=self._get_value('implementation/container/conda/content'))

    @property
    def command(self):
        return self._get_value('implementation/container/command')

    @property
    def args(self):
        def handle_placeholder(value):
            if isinstance(value, str):
                return value
            elif isinstance(value, dict):
                input_port_name = value.get('inputValue')
                if input_port_name:
                    port = self.inputs.get(input_port_name)
                    if not port:
                        print(f"Input port '{input_port_name}' not defined.")
                        return None
                    return port.value

                output_port_name = value.get('outputPath')
                if output_port_name:
                    port = self.outputs.get(output_port_name)
                    if not port:
                        raise ValueError(f"Output port '{output_port_name}' not defined.")
                    return port.value

                raise ValueError(f"'inputValue' or 'outputPath' must be specified in placeholder: {value}")
            else:
                raise ValueError(f"Incorrect type for arg {value}")

        raw_args = self._get_value('implementation/container/args')
        args = []
        for raw_arg in raw_args:
            arg = handle_placeholder(raw_arg)
            # For connected or assigned arguments, add to command arguments
            # Otherwise delete the argument name from command argument flag
            if arg:
                args.append(arg)
            else:
                args.pop()
        return args

    @property
    def command_and_args(self):
        return self.command + self.args

    @property
    def source_directory(self):
        return self._src_dir


class PipelineStep(PythonScriptStep):
    SCRIPT_FILE_NAME = 'invoker.py'

    def __init__(self, module, allow_reuse=True, run_config=None):
        self._comp = module

        if not run_config:
            run_config = self.get_default_run_config()

        self._script_path = self._copy_script_to_src_dir()

        print(self._comp.source_directory)
        print(self._script_path)

        print(f"== Creating KubeflowComponentStep: name={self._comp.name}\n"
              f"   arguments={self._comp.command_and_args}\n"
              f"   inputs={self._comp.input_refs}\n"
              f"   outputs={self._comp.output_refs}\n"
              )

        super().__init__(
            name=self._comp.name,
            source_directory=self._comp.source_directory,
            script_name=self._script_path,
            arguments=self._comp.command_and_args,
            inputs=self._comp.input_refs,
            outputs=self._comp.output_refs,
            compute_target=run_config.target,
            allow_reuse=allow_reuse,
            runconfig=run_config
        )

    def get_default_run_config(self):
        if self._comp.image:
            run_config = RunConfiguration()
            run_config.environment.docker.base_image = self._comp.image
        else:
            run_config = RunConfiguration(conda_dependencies=self._comp.conda_dependencies)
        run_config.target = DefaultConstants.COMPUTE_NAME
        run_config.environment.docker.enabled = True

        return run_config

    def _copy_script_to_src_dir(self):
        # new_script_name = f'{self.SCRIPT_FILE_NAME[:-3]}_{str(uuid.uuid1()).replace("-", "")}.py'
        new_script_name = self.SCRIPT_FILE_NAME
        dest_script_path = os.path.join(self._comp.source_directory, new_script_name)
        src_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.SCRIPT_FILE_NAME)
        print(f"Copy file {src_script_path} to {dest_script_path}")
        shutil.copyfile(src_script_path, dest_script_path)
        return new_script_name

    @property
    def script_name(self):
        return self._script_path


def run_pipeline(steps, experiment_name, workspace=None):
    if not workspace:
        workspace = get_default_workspace()

    exp = Experiment(workspace=workspace, name=experiment_name)
    pipeline = Pipeline(workspace=workspace, steps=steps)
    pipeline.validate()

    run = exp.submit(pipeline)
    run.wait_for_completion(show_output=True)
    run.get_metrics()
