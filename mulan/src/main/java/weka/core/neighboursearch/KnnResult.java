package weka.core.neighboursearch;

import weka.core.Instances;

/**
 * To store the kNNs and their indices, which would be used for LinearNNSearch2
 * 
 * @author Bin Liu
 *
 */

public class KnnResult{
	public Instances knn;
	public int[] indices;
	
	public KnnResult(Instances knn, int[] indices) {
		this.knn=knn;
		this.indices=indices;
	}
}