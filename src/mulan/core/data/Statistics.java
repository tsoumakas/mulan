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

/*
*    Statistics.java
*    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
*/
package mulan.core.data;

import mulan.core.*;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Set;
import mulan.core.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;

/**
 <!-- globalinfo-start -->
 * Class for calculating statistics of a multilabel dataset <p>
 * <br/>
 * For more information, see<br/>
 * <br/>
 * G. Tsoumakas, I. Katakis (2007). Multi-Label Classification: An Overview. International Journal of Data Warehousing and Mining, 3(3):1-13.
 * </p>
 <!-- globalinfo-end -->
 * 
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{tsoumakas+katakis:2007,
 *    author = {G. Tsoumakas, I. Katakis},
 *    journal = {International Journal of Data Warehousing and Mining},
 *    pages = {1-13},
 *    title = {Multi-Label Classification: An Overview},
 *    volume = {3},
 *    number = {3},
 *    year = {2007}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -F &lt;filename&gt;
 *  The filename (including full path) of the multilabel mlData set).</pre>
 * 
 * <pre> -L &lt;number of labels&gt;
 *  Number of labels. </pre>
 * 
 <!-- options-end -->
 *
 * @author Grigorios Tsoumakas 
 * @author Robert Friberg
 * @version $Revision: 0.03 $ 
 */
public class Statistics implements Serializable
{    
    private static final long serialVersionUID = 1206845794397561633L;

    /** the number of instances */
    private int numInstances;
    
    /** the number of predictive attributes */
    private int numPredictors = 0;
    
    /** the number of nominal predictive attributes */
    private int numNominal = 0;
    
    /** the number of numeric attributes */
    private int numNumeric = 0;
    
    /** the number of labels */
    private int numLabels;
    
    /** the label density  */
    private double labelDensity;

    /** the label cardinality */
    private double labelCardinality;
        
    /** percentage of instances per label */
    private double[] examplesPerLabel;
    
    /** number of examples per cardinality, <br><br>
     *  note that this array has size equal to the number of elements plus one, <br>
     *  because the first element is the number of examples for cardinality=0  */
    private double[] cardinalityDistribution;
    
    /** labelsets and their frequency */
    private HashMap<LabelSet,Integer> labelsets;
    
        
    public Statistics() { }
    
    /** 
     * returns the HashMap containing the distinct labelsets and their frequencies
     */
    public HashMap<LabelSet,Integer> labelCombCount() {
    	return labelsets;
    }
    
    
    /** 
     * This method calculates and prints a matrix with the coocurrences of <br>
     * pairs of labels 
     */
    public double[][] calculateCoocurrence(Instances data, int labels) {
        double[][] coocurrenceMatrix = new double[labels][labels];
        
        numPredictors = data.numAttributes()-labels;
        for (int k=0; k<data.numInstances(); k++) {
            Instance temp = data.instance(k);
            for (int i=0; i<labels; i++)
                for (int j=0; j<labels; j++) {
                    if (i >= j)
                        continue;
                    if (data.attribute(numPredictors+i).value((int) temp.value(numPredictors+i)).compareTo("1") == 0 && 
                        data.attribute(numPredictors+j).value((int) temp.value(numPredictors+j)).compareTo("1") == 0 )
                        coocurrenceMatrix[i][j]++;
                }
        }
        
        for (int i=0; i<labels; i++) {
            for (int j=0; j<labels; j++) {
                System.out.print(coocurrenceMatrix[i][j] + "\t");
            }
            System.out.println();
        }
        
        return coocurrenceMatrix;
    }    
       
    /** 
     * calculates various multilabel statistics, such as label cardinality, <br>
     * label density and the set of distinct labels along with their frequency
     */
    public void calculateStats(MultiLabelInstances mlData)
    {
        // initialize statistics
        Instances data = mlData.getDataSet();
        numLabels = mlData.getNumLabels();
        int[] labelIndices = mlData.getLabelIndices();
        int[] featureIndices = mlData.getFeatureIndices();
        numPredictors = featureIndices.length;

        labelCardinality = 0;
        numNominal = 0;
        numNumeric = 0;
        examplesPerLabel = new double[numLabels];
        cardinalityDistribution = new double[numLabels+1];
        labelsets = new HashMap<LabelSet,Integer>();
             
        // gather statistics
        for (int i=0; i<featureIndices.length; i++) {
            if (data.attribute(featureIndices[i]).isNominal())
                numNominal++;
            if (data.attribute(featureIndices[i]).isNumeric())
                numNumeric++;
        }        
        
        numInstances = data.numInstances();
        for (int i=0; i<numInstances; i++)
        {
            int exampleCardinality=0;
            double[] dblLabels = new double[numLabels];
            for (int j=0; j<numLabels; j++)
            {
            	double value = Double.parseDouble(data.attribute(labelIndices[j]).value((int) data.instance(i).value(labelIndices[j])));
                
                if (data.attribute(labelIndices[j]).value((int) value).equals("1"))
                {
                    dblLabels[j] = 1;
                    exampleCardinality++;
                    labelCardinality++;
                    examplesPerLabel[j]++;
                } else
                    dblLabels[j] = 0;
            }
            cardinalityDistribution[exampleCardinality]++;
            
            LabelSet labelSet = new LabelSet(dblLabels);
            if (labelsets.containsKey(labelSet))
                labelsets.put(labelSet, labelsets.get(labelSet) + 1);
            else
                labelsets.put(labelSet, 1);
        }
        
        labelCardinality /= numInstances;
        labelDensity = labelCardinality / numLabels;
        for (int j=0; j<numLabels; j++)
            examplesPerLabel[j] /= numInstances;        
    }
    
    
    /** 
     * returns various multilabel statistics in textual representation 
     */
    @Override
    public String toString() {
        String description = "";
        
        description += "Examples: " + numInstances + "\n";
        description += "Predictors: " + numPredictors + "\n";
        description += "--Nominal: " + numNominal + "\n";
        description += "--Numeric: " + numNumeric + "\n";
        
        description += "Labels: " + numLabels + "\n";

        description += "\n";       
        description += "Cardinality: " + labelCardinality + "\n";
        description += "Density: " + labelDensity + "\n";
        description += "Distinct Labelsets: " + labelsets.size() +"\n";
        
        description += "\n";
        for (int j=0; j<numLabels; j++)
            description += "Percentage of examples with label " + (j+1) + ": " + examplesPerLabel[j] + "\n";        
        
        description += "\n";
        for (int j=0; j<=numLabels; j++)
            description += "Examples of cardinality " + j + ": " + cardinalityDistribution[j] + "\n";            	

        description += "\n";
        for(LabelSet set : labelsets.keySet())
            description += "Examples of combination " + set + ": " +	labelsets.get(set) + "\n";

        return description;
    }
           
    /** 
     * returns the prior probabilities of the labels
     */
    public double[] priors() {
        double[] pr = new double[numLabels];
        for (int i=0; i<numLabels; i++)
            pr[i] = examplesPerLabel[i]/numInstances;
        return pr;
    }

    /** 
     * returns the label cardinality of the dataset
     */
    public double cardinality() {
        return labelCardinality;
    }    
    
    /** 
     * returns the label density of the dataset
     */
    public double density() {
        return labelDensity;
    }

    /** 
     * returns a set with the distinct labelsets of the dataset
     */
    public Set<LabelSet> labelSets() {
        return labelsets.keySet();
    }
    
    /** 
     * returns the frequency of a labelset in the dataset
     */
    public int labelFrequency(LabelSet x) {
        return labelsets.get(x);
    }
    
    /**
     * Estimates the number of labels by counting the number of binary <br>
     * attributes at the end of the attribute list
     * 
     * @param mlData
     * @return the number of labels in the dataset
     */
    public static int guessLabels(Instances data) 
    {
        int numAtts = data.numAttributes();
        int labels=0;
        for (int i=numAtts-1; i>=0; i--)
            if (data.attribute(i).isNominal() && data.attribute(i).numValues() == 2)
                labels++;
            else
                break;
        return labels;
    }
}

