package mulan.classifier;

import java.util.List;

import sun.reflect.generics.reflectiveObjects.NotImplementedException;

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
		// creates immutable instance where confidences are null 
	}
	
	/**
	 * Creates a new instance.
	 * @param ranks
	 * @param confidences
	 */
	public BipartitionAndRanking(List<Double> bipartitions, List<Integer> ranks, List<Double> confidences){
		this(bipartitions, ranks);
		// creates immutable instance 
	}
	
	public List<Boolean> getBipartitions(){
		throw new NotImplementedException();
	}
	
	public List<Integer> getRanks(){
		throw new NotImplementedException();
	}
	
	public List<Double> getConfidences(){
		throw new NotImplementedException();
	}
	
}
