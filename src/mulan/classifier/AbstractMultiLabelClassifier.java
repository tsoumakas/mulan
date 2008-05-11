package mulan.classifier;

/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

import java.util.ArrayList;
import java.util.Date;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;


/**
 * Base class of all MultiLabel classifiers.
 * Wrapper class for a single (or ensemble) binary classifer used on 
 * the transformed data.
 *
 * @author Robert Friberg
 * @version $Revision: 0.02 $ 
*/
public  abstract class AbstractMultiLabelClassifier 
extends Classifier 
implements TechnicalInformationHandler, MultiLabelClassifier
{

	/**
	 * Labels with probability over this value are included in 
	 * the output unless each label has an individual threshold,
	 *  see thresholds[].
	 */
	protected double threshold = 0.5;
	
	/**
	 *  Individual thresholds for each label.
	 */
	private double[] thresholds;
	
	/**
	 * All multilabel classifiers have this property per definition.
	 */
	protected int numLabels;
	
	
	/**
	 * The encapsulated classifier or used for making clones in the 
	 * case of ensemble classifiers. 
	 */
	protected Classifier baseClassifier;
	
	
	public enum SubsetMappingMethod
	{
		NONE,
		GREEDY,
		PROBABILISTIC
	}

	protected SubsetMapper subsetMapper;
	protected HybridSubsetMapper hybridMapper;
	private SubsetMappingMethod subsetMappingMethod;
	private int subsetDistanceThreshold = -1;
	
	
	public AbstractMultiLabelClassifier(){}
	
	
	public AbstractMultiLabelClassifier(int numLabels)
	{
		this.numLabels = numLabels;
	}
	

	protected void dbg(String msg)
	{
		if (!getDebug()) return;
		System.err.println("" + new Date() + ": " + msg);
	}
	
	public int getNumLabels()
	{
		return numLabels;
	}

	public void setNumLabels(int numLabels){
		this.numLabels = numLabels;
	}
	
	public TechnicalInformation getTechnicalInformation()
	{
            TechnicalInformation result = new TechnicalInformation(Type.ARTICLE);
            result.setValue(Field.AUTHOR, "Grigorios Tsoumakas, Ioannis Vlahavas");
            result.setValue(Field.YEAR, "2007");
            result.setValue(Field.TITLE, "Multi-Label Classification: An Overview");
            result.setValue(Field.JOURNAL, "International Journal of Data Warehousing and Mining");
            result.setValue(Field.VOLUME, "3(3)");
            result.setValue(Field.PAGES, "1-13");
            return result;
	}
	
	@Override
	public void buildClassifier(Instances instances) throws Exception
	{
		//if (subsetMappingMethod == SubsetMappingMethod.GREEDY)
		//{
			dbg("Building SubsetMapper");
			//subsetMapper = subsetDistanceThreshold > 0 ?
			//		new SubsetMapper(instances, numLabels, subsetDistanceThreshold) :
				subsetMapper = new SubsetMapper(instances, numLabels);
		//}
		//else if (subsetMappingMethod == SubsetMappingMethod.PROBABILISTIC)
		//{
			dbg("Building HybridSubsetMapper");
			hybridMapper = //subsetDistanceThreshold > 0 ?
					new HybridSubsetMapper(instances, numLabels, 4);
						//new HybridSubsetMapper(instances, numLabels);
		
			
	}

	/**
	 * Must override distribution for instance in subclass. distribution is a bad name
	 * for multilabel classification because the values are unrelated representing the
	 * confidence in the predicted    
	 */
	public double[] distributionForInstance(Instance instance) throws Exception {
		throw new Exception("Must override distribution for instance in subclass");

	}

	/**
	 * Multilabel classifiers cant return a single value, 
	 * throws Exception unconditionally. Cant be redefined in subclass. 
	 * @return never
	 * @throws Exception
	 */
	public final double classifyInstance() throws Exception
	{
		throw new Exception("Multilabel classifiers cant return a single value");
	}

	/**
	 *  The default base classifier if none is set before buildClassifier is called.
	 */
	public Classifier defaultClassifier()
	{
		return new weka.classifiers.trees.J48();
	}

		
	/**
	 * Template method enforcing application of useNearestSubset feature
	 * and possible future common logic. Subclass must override makePrediction
	 * which provides input to this method.
	 */
	public final Prediction predict(Instance instance) throws Exception
	{
		Prediction original = makePrediction(instance);
		if (subsetMappingMethod == SubsetMappingMethod.GREEDY)
		{
			return subsetMapper.nearestSubset(instance, original.predictedLabels);
		}
		else if (subsetMappingMethod == SubsetMappingMethod.PROBABILISTIC)
		{
			return hybridMapper.nearestSubset(instance, original.predictedLabels);
		}
		else return original;
	}
	
	protected abstract Prediction makePrediction(Instance instance) throws Exception;

	
	/**
	 * Derive output labels from distribution. Override in subclasses to 
	 * alter default behavior.
	 */
	protected double[] labelsFromConfidences(double[] confidences)
	{
		if (thresholds == null)
		{
			thresholds = new double[numLabels];
			java.util.Arrays.fill(thresholds, threshold);
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
	
	public int RandomIndexOfMax(double array[], Random rand) {
            double max = array[0];
            Vector indeces = new Vector();
            indeces.add(0);

            ArrayList<Integer> indexes = new ArrayList<Integer>();
            
            for (int i = 1; i < array.length; i++) {
                if (array[i] == max) {
                    indeces.add(i);
                } else
                if (array[i] > max) {
                    indeces = new Vector();
                    indeces.add(i);
                    max = array[i];
                } 
            }

            int numIndeces = indeces.size();
            if (numIndeces == 1)
                return ((Integer) indeces.get(0));           
            else
            {
                int Choose = rand.nextInt(indeces.size());
                return ((Integer) indeces.get(Choose));
            }
	}

	public Classifier getBaseClassifier()
	{
		return baseClassifier;
	}

	public void setBaseClassifier(Classifier baseClassifier)
	{
		this.baseClassifier = baseClassifier;
	}


	public void setThreshold(double threshold)
	{
		this.threshold = threshold;
	}


	public double getThreshold()
	{
		return threshold;
	}


	public void setThresholds(double[] thresholds)
	{
		this.thresholds = thresholds;
	}


	public double[] getThresholds()
	{
		return thresholds;
	}
	
	public void setOptions(String[] options) throws Exception
	{
	    
	    String method = Utils.getOption("-nearest-subset-method", options);
	    if (method.length() > 0)
	    {
	    	if (method.equals("greedy")) subsetMappingMethod = SubsetMappingMethod.GREEDY;
	    	else if (method.equals("prob")) subsetMappingMethod = SubsetMappingMethod.PROBABILISTIC;
	    	else throw new Exception("Invalid option to --nearest-subset-method: " + method);
	    	String strThreshold = Utils.getOption("-diff", options);
	    	if (strThreshold.length() > 0) subsetDistanceThreshold = Integer.parseInt(strThreshold);
	    	
	    }
	    super.setOptions(options);
	    Utils.checkForRemainingOptions(options);
	}
	
	
	public String[] getOptions()
	{
		Vector<String> result = new Vector<String>();
		if (subsetMappingMethod == SubsetMappingMethod.GREEDY)
		{
			result.add("--nearest-subset-method");
			result.add("greedy");
		}
		else if (subsetMappingMethod == SubsetMappingMethod.PROBABILISTIC)
		{
			result.add("--nearest-subset-method");
			result.add("prob");
		}

		if (subsetDistanceThreshold > 0 && 
				subsetMappingMethod != SubsetMappingMethod.NONE)
		{
			result.add("--diff");
			result.add("" + subsetDistanceThreshold);
		}
		if (getDebug()) result.add("-D");	
		
		return result.toArray(new String[0]);
	}


	public void setSubsetDistanceThreshold(int subsetDistanceThreshold)
	{
		this.subsetDistanceThreshold = subsetDistanceThreshold;
	}


	public int getSubsetDistanceThreshold()
	{
		return subsetDistanceThreshold;
	}


	public void setSubsetMethod(SubsetMappingMethod mm)
	{
		subsetMappingMethod = mm;
	}
}
