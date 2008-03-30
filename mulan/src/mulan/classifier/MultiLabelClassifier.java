package mulan.classifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public interface MultiLabelClassifier
{

	public int getNumLabels();
	
	public void setNumLabels(int numLabels);
	
	public void setBaseClassifier(Classifier classifier);
	
	public Classifier getBaseClassifier();
	

	/**
	 * What about the name predict? should we call it classify? 
	 * predictInstance? makePrediction?
	 * @param instance
	 * @return
	 * @throws Exception
	 */
	public Prediction predict(Instance instance) throws Exception;
	
	public void buildClassifier(Instances instances) throws Exception;

}