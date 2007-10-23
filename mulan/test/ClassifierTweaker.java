import mulan.classifier.MultiLabelClassifier;
/**
 * Emergency hack to the Experiment class allowing parameters to be
 * changed without retraining.
 * @author rofr
 *
 */
public interface ClassifierTweaker
{
	/**
	 * Called from within the experiment class, alter state of the
	 * passed classifier and return true to repeat a variation
	 * of the current evaluation.
	 * @param classifier 
	 * @param step allows you to distinguish between calls.
	 * @return true to run a new evaluation, false to notify 
	 * evaluation is completed with this classifier. 
	 */
	boolean tweak(MultiLabelClassifier classifier, int step);
}
