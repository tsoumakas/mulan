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
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;

import mulan.core.LabelSet;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

/**
 * Class that implements the PPT algorithm <p>
 *
 * @author Elise Rairat 
 * @author Grigorios Tsoumakas 
 * @version $Revision: 0.4 $
 */
public class PPT extends LabelPowerset {
    
    /*parameter for the threshold of number of occurences of a labelset */
    protected int x;
    
    /*parameter for the threshold of number of occurences of a labelset */
    protected boolean informationLoss=true;
            
    /** labelsets and their frequency of all label*/
    private HashMap<LabelSet,Integer> labelsets = new HashMap<LabelSet,Integer>();;          

    /** 
    * @paramater:
    * @param x: number of instances required for a labelset to be included.
    */
    public PPT(Classifier classifier, int numLabels, int x) throws Exception
    {
        super(classifier, numLabels);
        this.x = x; // x should be larger than 0
        setMakePredictionsBasedOnConfidences(true);
        threshold = 0.21;
    }

    /**
     * @param b true/false value for information loss
     */
    public void setInformationLoss(boolean b)
    {
        informationLoss = b;
    }

    /**
    * Returns an instance of a TechnicalInformation object, containing 
    * detailed information about the technical background of this class,
    * e.g., paper reference or book this class is based on.
    * 
    * @return the technical information about this class
    */
    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.INPROCEEDINGS);
        result.setValue(Field.AUTHOR, "Read, Jesse");
        result.setValue(Field.TITLE, "A Pruned Problem Transformation Method for Multi-label classification");
        result.setValue(Field.PAGES, "143-150");
        result.setValue(Field.BOOKTITLE, "Proc. 2008 New Zealand Computer Science Research Student Conference (NZCSRS 2008)");
        result.setValue(Field.YEAR, "2008");

        return result;
    }    
    
    @Override
    public void buildClassifier(Instances data) throws Exception
    {   
        int numInstances = data.numInstances();
        int numPredictors = data.numAttributes()-numLabels;
        metadataTrain = new Instances(data, 0);
        
        // create a data structure that holds for each labelset a list with the 
        // corresponding instances
        HashMap<LabelSet,ArrayList<Instance>> ListInstancePerLabel = new HashMap<LabelSet,ArrayList<Instance>>();
        for (int i=0; i<numInstances; i++)
        {
            double[] dblLabels = new double[numLabels];
            for (int j=0; j<numLabels; j++)
            {
            	double value = Double.parseDouble(data.attribute(numPredictors+j).value((int) data.instance(i).value(numPredictors + j))); 
                dblLabels[j] = value;                 
            }
            
            LabelSet labelSet = new LabelSet(dblLabels);
            if (labelsets.containsKey(labelSet))
                labelsets.put(labelSet, labelsets.get(labelSet) + 1);
            else 
                labelsets.put(labelSet, 1);
            
            if (ListInstancePerLabel.containsKey(labelSet))
                ListInstancePerLabel.get(labelSet).add(data.instance(i));
            else {
               ArrayList<Instance> li = new ArrayList<Instance>();
               li.add(data.instance(i));
               ListInstancePerLabel.put(labelSet, li);
            }                
        }
        
        // iterate the structure and a) if occurences of a labelset are higher 
        // than paramater then add them to the training set, b) if occurences
        // are less, then depending on the parameter either discard or create
        // new instances
        Instances newData = new Instances(data, 0);
        Iterator<LabelSet> it = ListInstancePerLabel.keySet().iterator(); 
        while(it.hasNext()) { 
            LabelSet ls = it.next(); 
            ArrayList<Instance> instances = ListInstancePerLabel.get(ls); 
            if (instances.size() > x)
                for (int i=0; i<instances.size(); i++)
                    newData.add(instances.get(i));
            else 
                if (!informationLoss) {
                    // split LabelSet into smaller ones
                    //System.out.println("original:" + ls.toString());
                    ArrayList<LabelSet> subsets = ls.getSubsets();
                    // sort subsets based on size
                    Collections.sort(subsets);
                    //for (LabelSet l: subsets) System.out.println(l.toString());
                    ArrayList<LabelSet> subsetsForInsertion = new ArrayList<LabelSet>();
                    for (LabelSet l: subsets)
                    {
                        //check if it exists in the training set
                        if (!ListInstancePerLabel.containsKey(l))
                            continue;
                        else 
                            // check if it has more than p elements
                            if (ListInstancePerLabel.get(l).size() <= x)
                                continue;
                            else {
                                // check that it has no common elements with 
                                // previously selected subsets
                                boolean foundCommon = false;
                                for (LabelSet l2: subsetsForInsertion)
                                {
                                    LabelSet temp = LabelSet.intersection(l, l2);
                                    if (temp.size() != 0) {
                                        foundCommon = true;
                                        break;
                                    }
                                }
                                if (foundCommon)
                                    continue;
                                else 
                                    subsetsForInsertion.add(l);
                            }
                    }
                    // insert subsetsForInsertion with corresponding instances
                    // from the original labelset
                    for (Instance tempInstance: instances) {
                        double[] temp = tempInstance.toDoubleArray();
                        for (LabelSet l: subsetsForInsertion) {
                            double[] tempLabels = l.toDoubleArray();                            
                            for (int i=0; i<numLabels; i++)
                                temp[numPredictors+i] = tempLabels[i];       
                            Instance newInstance = new Instance(1, temp);
                            newData.add(newInstance);
                        }
                    }
                }            
        }
                       
        super.buildClassifier(newData);
    }      
        
}
