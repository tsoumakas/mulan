package mulan.classifier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Represents a ranking of labels with respective confidences, provided by {@link MultiLabelRanker}. 
 * Confidences are optional.
 * 
 * @author Jozef Vilcek
 */
public class Ranking {
	
	private List<Integer> ranks;
	private List<Double> confidences;
	
	/**
	 * Creates a new instance.
	 * @param ranks
	 */
	public Ranking(List<Integer> ranks) {
		this(ranks, null);
	}
	
	/**
	 * Creates a new instance.
	 * @param ranks
	 */
	public Ranking(Integer[] ranks) {
		this(ranks, null);
	}
	
	/**
	 * Creates a new instance.
	 * @param ranks
	 */
	public Ranking(Integer[] ranks, Double[] confidences) {
		if(ranks == null){
			throw new IllegalArgumentException("Parameter ranks is null.");
		}
		if(confidences != null){
			if(ranks.length != confidences.length){
				throw new IllegalArgumentException("The dimension of ranks and confidences are not same.");
			}
			this.confidences = new ArrayList<Double>(Arrays.asList(confidences));
		}
		
		this.ranks = new ArrayList<Integer>(Arrays.asList(ranks));
	}
	
	/**
	 * Creates a new instance.
	 * @param ranks ranks for labels 
	 * @param confidences labels confidences
	 */
	public Ranking(List<Integer> ranks, List<Double> confidences){
		if(ranks == null){
			throw new IllegalArgumentException("Parameter ranks is null.");
		}
		if(confidences != null){
			if(ranks.size() != confidences.size()){
				throw new IllegalArgumentException("The dimension of ranks and confidences are not same.");
			}
			this.confidences = new ArrayList<Double>(confidences);
		}
		
		this.ranks = new ArrayList<Integer>(ranks);
	}
	
	/**
	 * Returns read-only list of ranks for labels, represented by indexes.
	 * 
	 * @return read-only list of ranks 
	 */
	public List<Integer> getRanks(){
		return Collections.unmodifiableList(ranks);
	}
	
	/**
	 * Returns read-only list of confidences for labels, represented by indexes.
	 * The confidences are optional because they are not available in all multi-label rankers.
	 * 
	 * @return read-only list of confidences or null if they are not available
	 */
	public List<Double> getConfidences(){
		return (confidences != null) ? Collections.unmodifiableList(confidences) : null;
	}
	
	

}
