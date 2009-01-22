package mulan.evaluation;

import java.util.List;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;


public class ExampleBasedMeasures {

	
	protected ExampleBasedMeasures(List<ModelEvaluationDataPair<Boolean>> learnerPredictionData){
		compute(learnerPredictionData);
	}
	
	protected ExampleBasedMeasures(ModelCrossValidationDataSet<Boolean> learnerCrossValPredictionDataSet){
		compute(learnerCrossValPredictionDataSet);
	}
	
	
	protected void compute(List<ModelEvaluationDataPair<Boolean>> learnerPredictionData){
		throw new NotImplementedException();
	}
	
	
	protected void compute(ModelCrossValidationDataSet<Boolean> learnerCrossValPredictionDataSet){
		throw new NotImplementedException();
	}
	
	public double getHammingLoss(){
		throw new NotImplementedException();
	}
	
	public double getSubsetAccuracy(){
		throw new NotImplementedException();
	}
	
	public double getAccuracy(){
		throw new NotImplementedException();
	}
	
	public double getFMeasure(){
		throw new NotImplementedException();
	}
	public double getPrecision(){
		throw new NotImplementedException();
	}
	
	public double getRecall(){
		throw new NotImplementedException();
	}
	
	//TODO: add more measures if applicable
	
	public String getAllMeasuresSummary(){
		throw new NotImplementedException();
	}
}
