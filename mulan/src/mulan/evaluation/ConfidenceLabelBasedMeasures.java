package mulan.evaluation;

import java.util.List;

import mulan.classifier.BipartitionAndRanking;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class ConfidenceLabelBasedMeasures {
	
	protected ConfidenceLabelBasedMeasures(List<ModelEvaluationDataPair<BipartitionAndRanking>> learnerPredictionData){
		// check if data contains confidences ... if do not this class does nothing
		compute(learnerPredictionData);
	}
	
	protected ConfidenceLabelBasedMeasures(ModelCrossValidationDataSet<BipartitionAndRanking> learnerCrossValPredictionDataSet){
		// check if data pairs in data set contains confidences ... if do not this class does nothing
		compute(learnerCrossValPredictionDataSet);
	}
	
	protected void compute(List<ModelEvaluationDataPair<BipartitionAndRanking>> learnerPredictionData){
		throw new NotImplementedException();
	}
	
	protected void compute(ModelCrossValidationDataSet<BipartitionAndRanking> learnerCrossValPredictionDataSet){
		throw new NotImplementedException();
	}
	
	public double getAUC(MeasureAveragingType averagingType){
		throw new NotImplementedException();
	}
	
	//TODO: add more measures if applicable
	
	public String getAllMeasuresSummary(){
		throw new NotImplementedException();
	}
	
	
}
