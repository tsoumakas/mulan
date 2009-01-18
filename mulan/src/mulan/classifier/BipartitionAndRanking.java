package mulan.classifier;

import java.util.List;

/**
 * Binary prediction (bipartition), ranking of labels and respective confidences. 
 * Confidences are optional.
 * 
 * @author Jozef Vilcek
 */
public class BipartitionAndRanking {

	/**
	 * Creates a new instance.
	 * @param ranks
	 */
	public BipartitionAndRanking(List<Double> bipartitions, List<Integer> ranks) {
		this(bipartitions, ranks, null);
		// convenience method
	}
	
	/**
	 * Creates a new instance.
	 * @param ranks
	 * @param confidences
	 */
	public BipartitionAndRanking(List<Double> bipartitions, List<Integer> ranks, List<Double> confidences){
		// creates immutable instance ... null value for bipartitions and ranks are not allowed
	}
	
	public List<Boolean> getBipartitions(){
		return null;
	}
	
	public List<Integer> getRanks(){
		return null;
	}
	
	public List<Double> getConfidences(){
		return null;
	}
	
}
