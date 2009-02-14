package mulan.classifier;

import java.util.List;

/**
 * Binary prediction (bipartition), ranking of labels and respective confidences. 
 * Confidences are optional.
 * 
 * @author Jozef Vilcek
 */
public class BipartitionAndRanking {

	private final Bipartition bipartition;
	private final Ranking ranking;
	
	/**
	 * Creates a new instance.
	 * @param bipartition
	 * @param ranks
	 */
	public BipartitionAndRanking(List<Boolean> bipartition, List<Integer> ranks) {
		this(bipartition, ranks, null);
	}
	
	/**
	 * Creates a new instance.
	 * @param bipartition
	 * @param ranks
	 * @param confidences
	 */
	public BipartitionAndRanking(List<Boolean> bipartition, List<Integer> ranks, List<Double> confidences){
		this.bipartition = new Bipartition(bipartition);
		this.ranking = new Ranking(ranks, confidences);
	}
	
	/**
	 * Creates a new instance.
	 * @param bipartition
	 * @param ranks
	 */
	public BipartitionAndRanking(Boolean[] bipartition, Integer[] ranks) {
		this(bipartition, ranks, null);
	}
	
	/**
	 * Creates a new instance.
	 * @param bipartition
	 * @param ranks
	 * @param confidences
	 */
	public BipartitionAndRanking(Boolean[] bipartition, Integer[] ranks, Double[] confidences){
		this.bipartition = new Bipartition(bipartition);
		this.ranking = new Ranking(ranks, confidences);
	}
	
	/**
	 * Creates a new instance.
	 * @param bipartition
	 * @param ranking
	 */
	public BipartitionAndRanking(Bipartition bipartition, Ranking ranking){
		this.bipartition = bipartition;
		this.ranking = ranking;
	}

	public List<Boolean> getBipartition(){
		return bipartition.getBipartition();
	}
	
	public List<Integer> getRanks(){
		return ranking.getRanks();
	}
	
	public List<Double> getConfidences(){
		return ranking.getConfidences();
	}
	
	public Bipartition getBipartitionObject(){
		return bipartition;
	}
	
	public Ranking getRankingObject(){
		return ranking;
	}
	
}
