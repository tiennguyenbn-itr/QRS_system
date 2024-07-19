import os
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

SHAPE = (145, 1)

TF_CONFIG = {
    'model_name': 'qrs',
    'signature': 'channels',
    'input': 'input',
    'output': 'prediction',
}

class ExportModel(tf.Module):
    def __init__(
            self,
            model
    ):
        super().__init__()
        self.model = model
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=(None, *SHAPE), dtype=tf.float32),
        ]
    )
    def score(
            self,
            input: tf.TensorSpec,
    ) -> dict:
        result = self.model([{
            TF_CONFIG['input']: input,
        }])
        return {
            TF_CONFIG['output']: result
        }
    
def export_model(model, output_path):
    os.makedirs(output_path, exist_ok=True)
    module = ExportModel(model)
    batched_module = tf.function(module.score)
    tf.saved_model.save(
        module,
        output_path,
        signatures={
            TF_CONFIG['signature']: batched_module.get_concrete_function(
                tf.TensorSpec(shape=(None, *SHAPE), dtype=tf.float32),
            )
        }
    )

def main(model_dir):
    model = tf.keras.models.load_model("model_213/model_run-0.h5")
    model_dir = f'model_new_2/2'
    os.makedirs(model_dir, exist_ok=True)
    export_model(model=model, output_path=model_dir)

if __name__ == '__main__':
    model_dir = 'model'
    main(model_dir)