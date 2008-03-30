package mulan.classifier;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

@SuppressWarnings("serial")
/**
 * A multilabel classifier based on Problem Transformation 5.
 * The multiple label attributes are mapped to a single multi class
 * attribute. 
 */
public class FlattenTrueLabelsClassifier extends AbstractMultiLabelClassifier implements
		MultiLabelClassifier
{

	public static final String versionId = "$Id: 2007-02-21 02:55:46 +0100 (on, 21 feb 2007) $"; 
	
	
	/**
	 * The encapsulated distribution classifier.
	 */
	protected Classifier classifier;
	
	
	/**
	 * A dataset with the format needed by the base classifier.
	 * It is potentially expensive copying datasets with many attributes,
	 * so it is used for building the classifier and then it's instances
	 * are discarded and it is reused during prediction.
	 */
	protected Instances transformed;

	
	/**
	 * Default constructor needed to allow instantiation 
	 * by reflection. If this constructor is used call setNumLabels()
	 * and setBaseClassifier(Classifier) before building the classifier
	 * or exceptions will hail.
	 */
	public FlattenTrueLabelsClassifier()
	{
	}
	
	public FlattenTrueLabelsClassifier(Classifier classifier, int numLabels)
	{
		super(numLabels);
		this.classifier = classifier;
	}
	
	public void buildClassifier(Instances instances) throws Exception
	{
		super.buildClassifier(instances);
		
		if (classifier == null) 
			classifier = forName("weka.classifiers.bayes.NaiveBayes", null);
		
		//Do the transformation 
		//and generate the classifier
		transformed = determineOutputFormat(instances); 
		transform(instances, transformed);
		classifier.buildClassifier(transformed);
		
		//We dont need the data anymore, just the metadata.
		//Asserts that the classifier has copied anything
		//it needs.
		transformed.delete();
	}
	

	/**
	 * 
	 * @param source
	 * @param dest
	 * @throws Exception
	 */
	protected void transform(Instances source, Instances dest) throws Exception
	{
		for (int i = 0; i < source.numInstances(); i++)
	    {
			//Convert a single instance to multiple ones
			Instance instance = source.instance(i);
			transformAndAppendMultiple(instance, dest);
	    }
	}
	
	/**
	 * Derives the transformed format suitable for the underlying classifier 
	 * from the input and the number of label attributes.
	 * @param input
	 * @return An empty Instances object with the new format
	 * @throws Exception
	 */
	protected Instances determineOutputFormat(Instances input) throws Exception 
	{
		//Get names of all class attributes
		FastVector classValues = new FastVector(numLabels);
		int startIndex = input.numAttributes() - numLabels; 
		for(int i = startIndex; i < input.numAttributes(); i++)
			classValues.addElement(input.attribute(i).name());
		
		//remove numClasses attributes from the end
		Instances outputFormat = new Instances(input, 0);
		for(int i = 0; i < numLabels; i++)
			outputFormat.deleteAttributeAt(outputFormat.numAttributes() - 1);
		
		//create and append the nominal class attribute
		Attribute classAttribute = new Attribute("Class", classValues);
		outputFormat.insertAttributeAt(classAttribute, outputFormat.numAttributes());
		outputFormat.setClassIndex(outputFormat.numAttributes() - 1);
		return outputFormat;
	}
	

	/**
	 * Each input instance will yield one output instance for each label.
	 * Transform a single input instance into 0 or more 
	 * output instances.
	 * 
	 * @param instance
	 * @param out Actually a reference to the member variable transformed.
	 */
	private void transformAndAppendMultiple(Instance instance, Instances out)
	{
		//Grab a reference to the input dataset 
		Instances in  = instance.dataset();
		
		//The prototype instance is used to make copies, one per label.
		Instance prototype = transform(instance);
	
		//At this point we have an instance with a missing last class value
	    //Now we iterate over the incoming instances multiple labels
	    //and output one instance for every label with a value of 1.
		//TODO: This asserts that label attribute is binary nominal with values {0,1}
		for(int i = in.numAttributes() - numLabels; i < in.numAttributes(); i++)
		{
			if (instance.value(i) == 0 || instance.value(i) == Instance.missingValue() ) continue;
			Instance copy = (Instance) prototype.copy();
			copy.setDataset(out);
			copy.setClassValue(instance.attribute(i).name());
			out.add(copy);
		}
	}
	
	/**
	 * Transform a single instance to match the format of the base
	 * classifier by copying all predictors to a new instance.
	 * @param instance
	 * @return a transformed copy of the passed instance
	 */
	private Instance transform(Instance instance)
	{
		
		//TODO: It might be faster to copy the entire instance and 
		//      then remove the trailing label attributes.		
		
		//Make room for all predictors and an additional class attribute
		double [] vals = new double[transformed.numAttributes()];

		//Copy all predictors		
		for (int i = 0; i < transformed.numAttributes() - 1; i++)
			vals[i] = instance.value(i);

		//If class value is 0 it will not be included in the sparse
		//instance and validation will fail. 
		vals[transformed.numAttributes()-1] = 42;
		
		Instance result = (instance instanceof SparseInstance)
		? new SparseInstance(instance.weight(), vals)
		: new Instance(instance.weight(), vals);
		
		result.setDataset(transformed);
		return result;
	}
	
	protected Prediction makePrediction(Instance instance) throws Exception
	{
		instance = transform(instance);
		double[] confidences = classifier.distributionForInstance(instance);
		return new Prediction(labelsFromConfidences(confidences), confidences);
	}
}
