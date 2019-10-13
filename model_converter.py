from keras.models import load_model
import coremltools

model.save('trainedModel.h5')

output_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
coreml_model = coremltools.converters.keras.convert('trainedModel.h5', input_names=['image'], output_names=['output'],
                                                   class_labels=output_labels, image_input_names='image')

coreml_model.author = 'Shreyash Nigam'
coreml_model.short_description = 'Sign language recognition'
coreml_model.input_description['image'] = 'Takes as input an image'
coreml_model.output_description['output'] = 'Prediction of Digit'

coreml_model.save('new_model.mlmodel')
