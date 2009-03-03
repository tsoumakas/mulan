package mulan.classifier.transformation;

import mulan.classifier.*;
import java.util.Arrays;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;

/**
 * A multilabel classifier based on Problem Transformation 6.
 * The multiple label attributes are mapped to two attributes:
 * a) one nominal attribute containing the class
 * b) one binary attribute containing whether it is true. 
 *
 * @author Robert Friberg
 * @author Grigorios Tsoumakas
 * @version $Revision: 0.03 $ 
 */
public class IncludeLabelsClassifier extends TransformationBasedMultiLabelLearner 
{
	double[] thresholds;
    double threshold=0.5;
	
	/**
	 * A dataset with the format needed by the base classifier.
	 * It is potentially expensive copying datasets with many attributes,
	 * so it is used for building the classifier and then it's instances
	 * are discarded and it is reused during prediction.
	 */
	protected Instances transformed;

	
	public IncludeLabelsClassifier(Classifier classifier, int numLabels)
	{
		super(classifier, numLabels);
	}
	
    @Override
    public void build(Instances instances) throws Exception
    {
            //Do the transformation 
            //and generate the classifier
            transformed = determineOutputFormat(instances);	
            transform(instances, transformed);

            /* debug info
            System.out.println(instances.instance(0).toString());
            for (int i=0; i<numLabels; i++)
                    System.out.println(transformed.instance(i).toString());
            //*/
            baseClassifier.buildClassifier(transformed);

            //We dont need the data anymore, just the metadata.
            //Asserts that the classifier has copied anything
            //it needs.
            transformed.delete();
            //System.out.println(transformed);
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
		
		//remove numLabels-1 attributes from the end
		Instances outputFormat = new Instances(input, 0);
		for(int i = 0; i < numLabels-1; i++)
			outputFormat.deleteAttributeAt(outputFormat.numAttributes() - 1);		
		
		//create and append the nominal class attribute before the end
		Attribute classAttribute = new Attribute("label", classValues);
		outputFormat.insertAttributeAt(classAttribute, outputFormat.numAttributes()-1);
		outputFormat.setClassIndex(outputFormat.numAttributes() - 1);
		return outputFormat;
	}
	

	/**
	 * Each input instance will yield one output instance for each label.
	 * 
	 * @param instance
	 * @param out Actually a reference to the member variable transformed.
	 */
	private void transformAndAppendMultiple(Instance instance, Instances out)
	{
		//Grab a reference to the input dataset 
		//Instances in  = instance.dataset();
		
		//The prototype instance is used to make copies, one per label.
	//	Instance prototype = transform(instance);
	
		for (int i=0; i<numLabels; i++) {
			Instance copy = (Instance) instance.copy();
			Instance newInstance = transform(copy, i);
			out.add(newInstance);
		}
		
	}
	
	/**
	 * Transform a single instance to match the format of the base
	 * classifier by copying all predictors to a new instance.
	 * @param instance
	 * @return a transformed copy of the passed instance
	 */
	private Instance transform(Instance instance, int label)
	{
		
		//TODO: It might be faster to copy the entire instance and 
		//      then remove the trailing label attributes.		
		
		//Make room for all predictors and an additional class attribute
		double [] vals = new double[transformed.numAttributes()];

		//Copy all predictors		
		for (int i = 0; i < transformed.numAttributes() - 2; i++)
			vals[i] = instance.value(i);

		vals[transformed.numAttributes()-2] = label;
		
		vals[transformed.numAttributes()-1] = instance.value(instance.numAttributes()-numLabels + label);
		
		Instance result = (instance instanceof SparseInstance)
		? new SparseInstance(instance.weight(), vals)
		: new Instance(instance.weight(), vals);
		
//		result.setDataset(transformed);
		return result;
	}
	/*
	public Bipartition predict(Instance instance) throws Exception
	{
		double[] confidences = new double[numLabels];
                //System.out.println(instance.toString());
                Instance newInstance;
		for (int i=0; i<numLabels; i++) {
			newInstance = transform(instance , i);	
                        newInstance.setDataset(transformed);
                        //System.out.println(instance.toString());
			double[] temp = baseClassifier.distributionForInstance(newInstance);
			confidences[i] = temp[transformed.classAttribute().indexOfValue("1")]; 
		}	
		//return new Prediction(labelsFromConfidences(confidences), confidences);
                return null;
	}*/

	/**
	 * Derive output labels from distributions.
	 */
	protected double[] labelsFromConfidences(double[] confidences)
	{
		if (thresholds == null)
		{
			thresholds = new double[numLabels];
			Arrays.fill(thresholds, threshold);
		}
		
		double[] result = new double[confidences.length];
		for(int i = 0; i < result.length; i++)
		{
			if (confidences[i] >= thresholds[i]){
				result[i] = 1.0;
			}
		}
		return result;
	}
	
    public String getRevision() {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public MultiLabelOutput makePrediction(Instance instance) throws Exception {
        throw new UnsupportedOperationException("Not supported yet.");
    }
	
}

