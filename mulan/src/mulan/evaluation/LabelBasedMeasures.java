package mulan.evaluation;

import java.util.List;

import mulan.classifier.Bipartition;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class LabelBasedMeasures {

	protected LabelBasedMeasures(List<ModelEvaluationDataPair<Bipartition>> predictionData,
								 ConfidenceLabelBasedMeasures confidenceLabelBasedMeasures){
		compute(predictionData);
	}
	
	protected LabelBasedMeasures(ModelCrossValidationDataSet<Bipartition> crossValPredictionDataSet,
								 ConfidenceLabelBasedMeasures confidenceLabelBasedMeasures){
		compute(crossValPredictionDataSet);
	}
	
	protected void compute(List<ModelEvaluationDataPair<Bipartition>> predictionData){
		throw new NotImplementedException();
	}
	
	protected void compute(ModelCrossValidationDataSet<Bipartition> crossValPredictionDataSet){
		throw new NotImplementedException();
	}
	
	public ConfidenceLabelBasedMeasures getConfidenceLabelBasedMeasures(){
		throw new NotImplementedException();
	}
	
	public double getAccuracy(MeasureAveragingType averagingType){
		throw new NotImplementedException();
	}
	
	public double getFMeasure(MeasureAveragingType averagingType){
		throw new NotImplementedException();
	}
	
	public double getRecall(MeasureAveragingType averagingType){
		throw new NotImplementedException();
	}
	
	//TODO: add more measures if applicable
	
	public String getAllMeasuresSummary(){
		throw new NotImplementedException();
	}
}
