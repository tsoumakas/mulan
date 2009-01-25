package mulan.classifier;


import weka.core.Instance;
import weka.core.Instances;

/**
 * Common interface for multi-label classifiers. 
 * 
 * @author Eleftherios Spyromitros-Xioufis ( espyromi@csd.auth.gr )
 * @author Jozef Vilcek
 */
public interface MultiLabelClassifier extends MultiLabelLearner
{
	/**
	 * Computes the binary prediction (bipartition) of labels for a 
	 * specified input {@link Instance}. The true for a label means that 
	 * the label is associated with input instance.
	 * The classifier model must be build before prediction as called.
	 * 
	 * @param instance the input instance for which the prediction is made
	 * @return the bipartition of labels
	 * @see MultiLabelLearner#build(Instances)
	 * @throws Exception if prediction was not successful
	 */
	public Bipartition predict(Instance instance) throws Exception;
}