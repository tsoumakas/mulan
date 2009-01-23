package mulan.evaluation;

import java.util.List;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class ConfidenceLabelBasedMeasures {
	
	protected ConfidenceLabelBasedMeasures(List<ModelEvaluationDataPair<Boolean>> learnerPredictionData){
		// check if data contains confidences ... if do not this class does nothing
		compute(learnerPredictionData);
	}
	
	protected ConfidenceLabelBasedMeasures(ModelCrossValidationDataSet<Boolean> learnerCrossValPredictionDataSet){
		// check if data pairs in data set contains confidences ... if do not this class does nothing
		compute(learnerCrossValPredictionDataSet);
	}
	
	protected void compute(List<ModelEvaluationDataPair<Boolean>> learnerPredictionData){
		throw new NotImplementedException();
	}
	
	protected void compute(ModelCrossValidationDataSet<Boolean> learnerCrossValPredictionDataSet){
		throw new NotImplementedException();
	}
	
	double getAUC(MeasureAveragingType averagingType){
		throw new NotImplementedException();
	}
	
	//TODO: add more measures if applicable
	
	public String getAllMeasuresSummary(){
		throw new NotImplementedException();
	}
	
	
}
