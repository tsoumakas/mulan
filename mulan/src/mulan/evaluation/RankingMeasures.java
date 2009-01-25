package mulan.evaluation;

import java.util.List;

import mulan.classifier.Ranking;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;


public class RankingMeasures {

	
	protected RankingMeasures(List<ModelEvaluationDataPair<Ranking>> rankingData){
		compute(rankingData);
	}
	
	protected RankingMeasures(ModelCrossValidationDataSet<Ranking> crossValRankingDataSet){
		compute(crossValRankingDataSet);
	}
	
	protected void compute(List<ModelEvaluationDataPair<Ranking>> rankingData){
		throw new NotImplementedException();
	}
	
	protected void compute(ModelCrossValidationDataSet<Ranking> crossValRankingDataSet){
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
