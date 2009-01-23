package mulan.evaluation;

import java.util.List;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class LabelBasedMeasures {

	protected LabelBasedMeasures(List<ModelEvaluationDataPair<Boolean>> learnerPredictionData){
		compute(learnerPredictionData);
	}
	
	protected LabelBasedMeasures(ModelCrossValidationDataSet<Boolean> learnerCrossValPredictionDataSet){
		compute(learnerCrossValPredictionDataSet);
	}
	
	protected void compute(List<ModelEvaluationDataPair<Boolean>> learnerPredictionData){
		throw new NotImplementedException();
	}
	
	
	protected void compute(ModelCrossValidationDataSet<Boolean> learnerCrossValPredictionDataSet){
		throw new NotImplementedException();
	}
	
	public ConfidenceLabelBasedMeasures getConfidenceLabelBasedMeasures(){
		throw new NotImplementedException();
	}
	
	double getAccuracy(MeasureAveragingType averagingType){
		throw new NotImplementedException();
	}
	
	double getFMeasure(MeasureAveragingType averagingType){
		throw new NotImplementedException();
	}
	
	double getRecall(MeasureAveragingType averagingType){
		throw new NotImplementedException();
	}
	
	//TODO: add more measures if applicable
	
	public String getAllMeasuresSummary(){
		throw new NotImplementedException();
	}
}
