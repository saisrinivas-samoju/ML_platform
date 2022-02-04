############################################## WAFER FAULT DETECTION ##############################################
from controller.project_controller.projects.wafer_fault_detection.training_Validation_Insertion import \
    TrainingValidation as TrainingValidationWafer
from controller.project_controller.projects.wafer_fault_detection.trainingModel import \
    TrainingModel as TrainModelWafer
from controller.project_controller.projects.wafer_fault_detection.prediction_Validation_Insertion import \
    PredictionValidation as PredictionValidationWafer
from controller.project_controller.projects.wafer_fault_detection.predictFromModel import \
    Prediction as PredictionWafer

############################################## CEMENT STRENGTH PREDICTION ##########################################
from controller.project_controller.projects.cement_strength.training_validation_insertion import TrainingValidation \
    as TrainingValidationCementStrength
from controller.project_controller.projects.cement_strength.training_model_cement_strength import TrainingModel \
    as TrainCementStrength
from controller.project_controller.projects.cement_strength.prediction_validation_insertion import  PredictionValidation \
    as PredictionValidationCementStrength
from controller.project_controller.projects.cement_strength.prediction_model_cement_strength import Prediction \
    as PredictCementStrength

############################################ CREDIT CARD DEFAULTER CLASSIFICATION ##################################
from controller.project_controller.projects.credit_card_default.training_validation_insertion import TrainingValidation \
    as TrainingValidationCreditDefaulters
from controller.project_controller.projects.credit_card_default.training_model_credit_deaulter import TrainingModel \
    as TrainCreditDefaulters
from controller.project_controller.projects.credit_card_default.prediction_validation_insertion import  PredictionValidation \
    as PredictionValidationCreditDefaulters
from controller.project_controller.projects.credit_card_default.prediction_model_credit_defaulter import Prediction \
    as PredictCreditDefaulters

############################################ FOREST COVER CLASSIFICATION ############################################
from controller.project_controller.projects.forest_cover_classification.training_validation_insertion import TrainingValidation \
    as TrainingValidationForesetCoverClassifier
from controller.project_controller.projects.forest_cover_classification.training_model_forest_cover import TrainingModel \
    as TrainForesetCoverClassifier
from controller.project_controller.projects.forest_cover_classification.prediction_validation_insertion import  PredictionValidation \
    as PredictionValidationForesetCoverClassifier
from controller.project_controller.projects.forest_cover_classification.prediction_model_forset_cover import Prediction \
    as PredictForesetCoverClassifier

############################################ PHISING CLASSIFIER ####################################################
from controller.project_controller.projects.phising_classifier.training_validation_insertion import TrainingValidation \
    as TrainingValidationPhisingClassifier
from controller.project_controller.projects.phising_classifier.training_model_phising_classifier import TrainingModel \
    as TrainPhisingClassifier
from controller.project_controller.projects.phising_classifier.prediction_validation_insertion import  PredictionValidation \
    as PredictionValidationPhisingClassifier
from controller.project_controller.projects.phising_classifier.prediction_model_phising_classifier import Prediction \
    as PredictPhisingClassifier

############################################ CLIMATE VISIBLITY ######################################################
from controller.project_controller.projects.visibility_climate.training_validation_insertion import TrainingValidation \
    as TrainingValidationVisibilityClimate
from controller.project_controller.projects.visibility_climate.train_model_visibility_climate import TrainingModel \
    as TrainVisibilityClimate
from controller.project_controller.projects.visibility_climate.prediction_validation_insertion import  PredictionValidation \
    as PredictionValidationVisibilityClimate
from controller.project_controller.projects.visibility_climate.prediction_model_visibility_climate import Prediction \
    as PredictVisibilityClimate

########################################## ENGINE FAULT DETECTION ###################################################
from controller.project_controller.projects.scania_truck.traning_validation_insertion import TrainingValidation \
    as TrainingValidationScaniaTruckPressure
from controller.project_controller.projects.scania_truck.training_model_scania_truck import TrainingModel \
    as TrainScaniaTruckPressure
from controller.project_controller.projects.scania_truck.prediction_validation_insertion import  PredictionValidation \
    as PredictionValidationScaniaTruckPressure
from controller.project_controller.projects.scania_truck.prediction_model_scania_truck import Prediction \
    as PredictScaniaTruckPressure


############# Project train and prediction mapping classes #############

project_train_and_prediction_mapper = [
    {
        'project_id': 1,
        'training_class_name': TrainModelWafer,
        'prediction_class_name': PredictionWafer,
        'training_validation_class_name': TrainingValidationWafer,
        'prediction_validation_class_name': PredictionValidationWafer
    },
    {
        'project_id': 2,
        'training_class_name': TrainCementStrength,
        'prediction_class_name': PredictCementStrength,
        'training_validation_class_name': TrainingValidationCementStrength,
        'prediction_validation_class_name': PredictionValidationCementStrength
    },
    {
        'project_id': 3,
        'training_class_name': TrainCreditDefaulters,
        'prediction_class_name': PredictCreditDefaulters,
        'training_validation_class_name': TrainingValidationCreditDefaulters,
        'prediction_validation_class_name': PredictionValidationCreditDefaulters
    },
    {
        'project_id': 4,
        'training_class_name': TrainForesetCoverClassifier,
        'prediction_class_name': PredictForesetCoverClassifier,
        'training_validation_class_name': TrainingValidationForesetCoverClassifier,
        'prediction_validation_class_name': PredictionValidationForesetCoverClassifier
    },
    {
        'project_id': 5,
        'training_class_name': TrainPhisingClassifier,
        'prediction_class_name': PredictPhisingClassifier,
        'training_validation_class_name': TrainingValidationPhisingClassifier,
        'prediction_validation_class_name': PredictionValidationPhisingClassifier
    },
    {
        'project_id': 6,
        'training_class_name': TrainVisibilityClimate,
        'prediction_class_name': PredictVisibilityClimate,
        'training_validation_class_name': TrainingValidationVisibilityClimate,
        'prediction_validation_class_name': PredictionValidationVisibilityClimate
    },
    {
        'project_id': 7,
        'training_class_name': TrainScaniaTruckPressure,
        'prediction_class_name': PredictScaniaTruckPressure,
        'training_validation_class_name': TrainingValidationScaniaTruckPressure,
        'prediction_validation_class_name': PredictionValidationScaniaTruckPressure
    },
]

def get_training_validation_and_training_model_class_name(project_id):
    try:
        for i in project_train_and_prediction_mapper:
            if i['project_id'] == project_id:
                return i['training_validation_class_name'], i['training_class_name'],

    except Exception as e:
        raise e


def get_prediction_validation_and_prediction_model_class_name(project_id):
    try:
        for i in project_train_and_prediction_mapper:
            if i['project_id'] == project_id:
                return i['prediction_validation_class_name'], i['prediction_class_name']
    except Exception as e:
        raise e
