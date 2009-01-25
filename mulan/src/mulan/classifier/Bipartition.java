package mulan.classifier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Binary prediction (bipartition) of labels.
 * 
 * @author Jozef Vilcek
 */
public class Bipartition {

	private List<Boolean> bipartition;
	
	/**
	 * Creates a new instance.
	 * @param bipartition labels bipartition
	 */
	public Bipartition(List<Boolean> bipartition) {
		if(bipartition == null){
			throw new IllegalArgumentException("Bipartition of labels is null.");
		}
		this.bipartition = new ArrayList<Boolean>(bipartition);
	}
	
	/**
	 * Creates a new instance.
	 * @param bipartition labels bipartition
	 */
	public Bipartition(Boolean[] bipartition) {
		if(bipartition == null){
			throw new IllegalArgumentException("Bipartition of labels is null.");
		}
		this.bipartition = new ArrayList<Boolean>(Arrays.asList(bipartition));
	}
	
	/**
	 * Returns a read-only list of labels bipartition.
	 * @return labels bipartition
	 */
	public List<Boolean> getBipartition(){
		return Collections.unmodifiableList(bipartition);
	}
}
