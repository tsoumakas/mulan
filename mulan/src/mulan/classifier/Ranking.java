package mulan.classifier;

import java.util.List;

import mulan.evaluation.ModelEvaluationDataPair;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

/**
 * Represents a ranking of labels with respective confidences, provided by {@link MultiLabelRanker}. 
 * Confidences are optional.
 * 
 * @author Jozef Vilcek
 */
public class Ranking {
	
	/**
	 * Creates a new instance.
	 * @param ranks
	 */
	public Ranking(List<Integer> ranks) {
		this(ranks, null);
		// convenience method
	}
	
	/**
	 * Creates a new instance.
	 * @param ranks
	 * @param confidences
	 */
	public Ranking(List<Integer> ranks, List<Double> confidences){
		// creates immutable instance ... null value for ranks is not allowed
	}
	
	/**
	 * Returns read-only list of ranks for labels, represented by indexes.
	 * 
	 * @return read-only list of ranks 
	 */
	public List<Integer> getRanks(){
		throw new NotImplementedException();
	}
	
	/**
	 * Returns read-only list of confidences for labels, represented by indexes.
	 * The confidences are optional because they are not available in all multi-label rankers.
	 * 
	 * @return read-only list of confidences or null if they are not available
	 */
	public List<Double> getConfidences(){
		throw new NotImplementedException();
	}
	
	

}
