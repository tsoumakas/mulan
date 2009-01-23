package mulan.evaluation;

import java.util.List;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;


public class RankingMeasures {

	
	protected RankingMeasures(List<ModelEvaluationDataPair<Boolean>> learnerRankingData){
		compute(learnerRankingData);
	}
	
	protected RankingMeasures(ModelCrossValidationDataSet<Boolean> learnerCrossValRankingDataSet){
		compute(learnerCrossValRankingDataSet);
	}
	
	protected void compute(List<ModelEvaluationDataPair<Boolean>> learnerRankingData){
		throw new NotImplementedException();
	}
	
	protected void compute(ModelCrossValidationDataSet<Boolean> learnerCrossValRankingDataSet){
		throw new NotImplementedException();
	}
	
	
	public double getAvgPrecision(){
		throw new NotImplementedException();
	}
	
	public double getRLoss(){
		throw new NotImplementedException();
	}
	
	public double getCoverage(){
		throw new NotImplementedException();
	}
	
	public double getOneError(){
		throw new NotImplementedException();
	}
	
	//TODO: add more measures if applicable
	
	public String getAllMeasuresSummary(){
		throw new NotImplementedException();
	}
}
