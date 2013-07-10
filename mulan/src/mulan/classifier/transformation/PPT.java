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
package mulan.classifier.transformation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.logging.Level;
import java.util.logging.Logger;
import mulan.data.DataUtils;
import mulan.data.LabelSet;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

/**
 * <p>Implementation of the Pruned Problem Transformation algorithm (PPT).</p>
 * <p>For more information, see <em> Read, J. (2008) A Pruned Problem
 * Transformation Method for Multi-label classification. In: Proc. 2008 New
 * Zealand Computer Science Research Student Conference (NZCSRS 2008), pp.
 * 143-150.</em></p> 
 * 
 * @author Grigorios Tsoumakas
 * @version 2012.02.27
 */
public class PPT extends LabelsetPruning {

    /**
     * strategies for processing infrequent labelsets
     */
    public enum Strategy {

        /**
         * Discard infrequent labelsets
         */
        INFORMATION_LOSS,
        /**
         * Reintroduce infrequent labelsets via subsets
         */
        NO_INFORMATION_LOSS;
    };
    /**
     * strategy for processing infrequent labelsets
     */
    private Strategy strategy;

    /**
     * Default constructor
     */
    public PPT() {
        this(new J48(), 3, Strategy.NO_INFORMATION_LOSS);
    }

    /**
     * Constructor that initializes learner with base algorithm, parameter p and
     * strategy for processing infrequent labelsets
     *
     * @param classifier base single-label classification algorithm
     * @param p number of instances required for a labelset to be included.
     * @param aStrategy strategy for processing infrequent labelsets
     */
    public PPT(Classifier classifier, int p, Strategy aStrategy) {
        super(classifier, p);
        strategy = aStrategy;
        setConfidenceCalculationMethod(2);
        setMakePredictionsBasedOnConfidences(true);
        threshold = 0.21;
    }

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

    /**
     * Returns a string describing classifier
     *
     * @return a description suitable for displaying
     */
    @Override
    public String globalInfo() {

        return "Class implementing the Pruned Problem Transformation "
                + "algorithm (PPT) . For more information, see\n\n"
                + getTechnicalInformation().toString();
    }

    @Override
    ArrayList<Instance> processRejected(LabelSet ls) {
        switch (strategy) {
            case INFORMATION_LOSS:
                return new ArrayList<Instance>();
            case NO_INFORMATION_LOSS:
                // split LabelSet into smaller ones
                //debug System.out.println("original:" + ls.toString());
                ArrayList<LabelSet> subsets = null;
                try {
                    subsets = ls.getSubsets();
                } catch (Exception ex) {
                    Logger.getLogger(PPT.class.getName()).log(Level.SEVERE, null, ex);
                }
                // sort subsets based on size
                Collections.sort(subsets);
                //debug for (LabelSet l: subsets) System.out.println(l.toString());
                ArrayList<LabelSet> subsetsForInsertion = new ArrayList<LabelSet>();
                for (LabelSet l : subsets) {
                    // check if it exists in the training set
                    if (!ListInstancePerLabel.containsKey(l)) {
                        continue;
                    }
                    // check if it occurs more than p times
                    if (ListInstancePerLabel.get(l).size() <= p) {
                        continue;
                    }
                    // check that it has no common elements with
                    // previously selected subsets
                    boolean foundCommon = false;
                    for (LabelSet l2 : subsetsForInsertion) {
                        LabelSet temp = LabelSet.intersection(l, l2);
                        if (temp.size() != 0) {
                            foundCommon = true;
                            break;
                        }
                    }
                    if (foundCommon) {
                        continue;
                    } else {
                        subsetsForInsertion.add(l);
                    }
                }

                // insert subsetsForInsertion with corresponding instances
                // from the original labelset
                ArrayList<Instance> instances = ListInstancePerLabel.get(ls);
                ArrayList<Instance> newInstances = new ArrayList<Instance>();
                for (Instance tempInstance : instances) {
                    for (LabelSet l : subsetsForInsertion) {
                        double[] temp = tempInstance.toDoubleArray();
                        double[] tempLabels = l.toDoubleArray();
                        for (int i = 0; i < numLabels; i++) {
                            if (format.attribute(labelIndices[i]).value(0).equals("0")) {
                                temp[labelIndices[i]] = tempLabels[i];
                            } else {
                                temp[labelIndices[i]] = 1 - tempLabels[i];
                            }
                        }
                        Instance newInstance = DataUtils.createInstance(tempInstance, 1, temp);
                        newInstances.add(newInstance);
                    }
                }
                return newInstances;
            default:
                return null;
        }
    }
}