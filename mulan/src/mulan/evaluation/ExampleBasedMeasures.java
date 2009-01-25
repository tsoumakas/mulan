package mulan.evaluation;

import java.util.List;

import mulan.classifier.Bipartition;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;


public class ExampleBasedMeasures {

	
	protected ExampleBasedMeasures(List<ModelEvaluationDataPair<Bipartition>> predictionData){
		compute(predictionData);
	}
	
	protected ExampleBasedMeasures(ModelCrossValidationDataSet<Bipartition> crossValPredictionDataSet){
		compute(crossValPredictionDataSet);
	}
	
	protected void compute(List<ModelEvaluationDataPair<Bipartition>> predictionData){
		throw new NotImplementedException();
	}
	
	protected void compute(ModelCrossValidationDataSet<Bipartition> crossValPredictionDataSet){
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
