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
 *    CalibratedLabelRanking.java
 *    Copyright (C) 2009 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */

package mulan.classifier.transformation;

import mulan.classifier.MultiLabelOutput;
import mulan.core.data.MultiLabelInstances;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Reorder;

/**
 *
 * <!-- globalinfo-start -->
 *
 * <pre>
 * Class implementing the Calibrated Label Ranking algorithm.
 * </pre>
 *
 * For more information:
 *
 * <pre>
 * Fürnkranz, J., Hüllermeier, E., Loza Mencía, E., and Brinker, K. (2008)
 * Multilabel classification via calibrated label ranking.
 * Machine Learning 73(2), 133-153
 * </pre>
 *
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start --> BibTeX:
 *
 * <pre>
 * &#064;article{furnkranze+etal:2008,
 *    author = {Fürnkranz, J. and Hüllermeier, E. and Loza Mencía, E. and Brinker, K.},
 *    title = {Multilabel classification via calibrated label ranking},
 *    journal = {Machine Learning},
 *    volume = {73},
 *    number = {2},
 *    year = {2008},
 *    pages = {133--153},
 * }
 * </pre>
 *
 * <p/> <!-- technical-bibtex-end -->
 *
 * @author Louise Rairat
 * @author Grigorios Tsoumakas
 * @version $Revision: 1.0 $
 */
public class CalibratedLabelRanking extends TransformationBasedMultiLabelLearner
{
     protected Instances[] dataset;
     protected Classifier[] oneVsOneModels;
     protected int numModels;
     protected int Matrix[][];
     protected Instances trainingdata;
     protected Instances[] metaDataTest;
     protected BinaryRelevance virtualLabelModels;
     
    public CalibratedLabelRanking(Classifier classifier) throws Exception 
    {
        super(classifier);
    }
    
    
    /**
     * This method deletes all the labels exepted the label in parameter.
     * @param instance
     * @param label
     * @return a instance with the attribute deleted
     * @throws java.lang.Exception
     */
    private Instance transformInstance(Instance instance, int label)
        throws Exception
    {
            Instance newInstance = new Instance(instance.numAttributes());
            newInstance = (Instance) instance.copy();
            newInstance.setDataset(null);
            int numPredictors = instance.numAttributes() - numLabels;
            int skipLabel = 0;
            for (int i = 0; i < numLabels; i++)
            {
                    if (i == label)
                    {
                            skipLabel++;
                            continue;
                    }
                    newInstance.deleteAttributeAt(numPredictors + skipLabel);
            }
            return newInstance;
    }
    

    @Override
    protected void buildInternal(MultiLabelInstances trainingSet) throws Exception
    {
        // Virtual label models
        debug("Building calibration label models");
        virtualLabelModels = new BinaryRelevance(getBaseClassifier());
        virtualLabelModels.build(trainingSet);

        // One-vs-one models
        numModels = ((numLabels)*(numLabels-1))/2;
        oneVsOneModels = Classifier.makeCopies(getBaseClassifier(), numModels);
        metaDataTest = new Instances[numModels];

        Instances trainingData = trainingSet.getDataSet();

        int counter=0;
        // Creation of one-vs-one models
        for (int label1=0; label1<numLabels-1; label1++)
        {
            // Attribute of label 1
            Attribute attrLabel1 = trainingData.attribute(labelIndices[label1]);
            for (int label2=label1+1; label2<numLabels; label2++)
            {
                debug("Building one-vs-one model " + (counter+1) + "/" + numModels);
                // Attribute of label 2
                Attribute attrLabel2 = trainingData.attribute(labelIndices[label2]);

                // initialize training set
                Instances dataOneVsOne = new Instances(trainingData, 0);

                // filter out examples with no preference
                for (int i=0; i<trainingData.numInstances(); i++)
                {
                    Instance tempInstance = new Instance(trainingData.instance(i));

                    int nominalValueIndex;
                    nominalValueIndex = (int) tempInstance.value(labelIndices[label1]);
                    String value1 = attrLabel1.value(nominalValueIndex);
                    nominalValueIndex = (int) tempInstance.value(labelIndices[label2]);
                    String value2 = attrLabel2.value(nominalValueIndex);

                    if (!value1.equals(value2)) {
                        tempInstance.setValue(attrLabel1, value1);
                        dataOneVsOne.add(tempInstance);
                    }
                }

                // remove all labels apart from label 1 and place it at the end
                Reorder filter = new Reorder();
                int numPredictors = trainingData.numAttributes()-numLabels;
                int[] reorderedIndices = new int[numPredictors+1];
                int labelIndicesCounter=0;
                int reorderedIndicesCounter=0;
                for (int i=0; i<numPredictors+numLabels; i++)
                {
                    if (labelIndices[labelIndicesCounter] == i)
                    {
                        labelIndicesCounter++;
                        continue;
                    }
                    reorderedIndices[reorderedIndicesCounter] = i;
                    reorderedIndicesCounter++;
                }
                reorderedIndices[reorderedIndicesCounter] = labelIndices[label1];
                filter.setAttributeIndicesArray(reorderedIndices);
                filter.setInputFormat(dataOneVsOne);
        		dataOneVsOne = Filter.useFilter(dataOneVsOne, filter);
                //System.out.println(dataOneVsOne.toString());
                dataOneVsOne.setClassIndex(numPredictors);

                // build model label1 vs label2
                oneVsOneModels[counter].buildClassifier(dataOneVsOne);
                dataOneVsOne.delete();
                metaDataTest[counter] = dataOneVsOne;
                counter++;
            }
        }
    }

    /**
     * This method does a prediction for an instance with the values of label missing
     * @param instance
     * @return prediction
     * @throws java.lang.Exception
     */
    public MultiLabelOutput makePrediction(Instance instance) throws Exception {
        boolean[] bipartition = new boolean[numLabels];
		double[] confidences = new double[numLabels];
        int[] voteLabel = new int[numLabels+1];

        // delete all labels and add a new atribute at the end
        Remove remove = new Remove();
        remove.setAttributeIndicesArray(labelIndices);
        remove.setInputFormat(instance.dataset());
        remove.setInvertSelection(false);
        Instance newInstance = null;
        if (remove.input(instance))
        {
            newInstance = remove.output();
        }
        newInstance.setDataset(null);
        newInstance.insertAttributeAt(numLabels);

        //initialize the array voteLabel
        for(int i=0; i<numLabels; i++) 
            voteLabel[i]=0;

        int counter=0;
        for (int label1=0; label1<numLabels-1; label1++)
        {
            for (int label2=label1+1; label2<numLabels; label2++)
            {
                double distribution[] = new double[2];
                try {
                    newInstance.setDataset(metaDataTest[counter]);
                    distribution = oneVsOneModels[counter].distributionForInstance(newInstance);
                } catch (Exception e) {
                    System.out.println(e);
                    return null;
                }
                int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;

                // Ensure correct predictions both for class values {0,1} and {1,0}
                Attribute classAttribute = metaDataTest[counter].classAttribute();

                if (classAttribute.value(maxIndex).equals("1"))
                    voteLabel[label1]++;
                else
                    voteLabel[label2]++;

                counter++;
            }

        }

        int voteVirtual=0;
        MultiLabelOutput virtualMLO = virtualLabelModels.makePrediction(instance);
        boolean[] virtualBipartition = virtualMLO.getBipartition();
        for (int i=0; i<numLabels; i++)
        {
            if (virtualBipartition[i])
                voteLabel[i]++;
            else
                voteVirtual++;
        }

        for(int i = 0; i<numLabels; i++)
        {
            if (voteLabel[i] >= voteVirtual)
                bipartition[i] = true;
            else 
                bipartition[i] = false;
            confidences[i]=1.0*voteLabel[i]/numLabels;
        }
        MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidences);
        return mlo;
    }
        
}
