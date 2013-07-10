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
import java.util.HashMap;
import java.util.Iterator;
import mulan.data.LabelSet;
import mulan.data.MultiLabelInstances;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 * <p>Common functionality class for the PPT and PS algorithms.</p>
 *
 * @author Grigorios Tsoumakas
 * @version 2012.02.27
 */
public abstract class LabelsetPruning extends LabelPowerset {

    /**
     * labelsets and a list with the corresponding instances
     */
    HashMap<LabelSet, ArrayList<Instance>> ListInstancePerLabel;
    /**
     * parameter for the threshold of number of occurences of a labelset
     */
    protected int p;
    /**
     * format of the data
     */
    Instances format;

    /**
     * Constructor that initializes learner with base algorithm and main
     * parameter
     *
     * @param classifier base single-label classification algorithm
     * @param aP number of instances required for a labelset to be included.
     */
    public LabelsetPruning(Classifier classifier, int aP) {
        super(classifier);
        if (aP <= 0) {
            throw new IllegalArgumentException("p should be larger than 0!");
        }
        p = aP;
        setConfidenceCalculationMethod(2);
        setMakePredictionsBasedOnConfidences(true);
        threshold = 0.21;
    }

    abstract ArrayList<Instance> processRejected(LabelSet ls);

    @Override
    protected void buildInternal(MultiLabelInstances mlDataSet) throws Exception {
        Instances data = mlDataSet.getDataSet();
        format = new Instances(data, 0);
        int numInstances = data.numInstances();

        ListInstancePerLabel = new HashMap<LabelSet, ArrayList<Instance>>();
        for (int i = 0; i < numInstances; i++) {
            double[] dblLabels = new double[numLabels];
            for (int j = 0; j < numLabels; j++) {
                int index = labelIndices[j];
                double value = Double.parseDouble(data.attribute(index).value((int) data.instance(i).value(index)));
                dblLabels[j] = value;
            }
            LabelSet labelSet = new LabelSet(dblLabels);
            if (ListInstancePerLabel.containsKey(labelSet)) {
                ListInstancePerLabel.get(labelSet).add(data.instance(i));
            } else {
                ArrayList<Instance> li = new ArrayList<Instance>();
                li.add(data.instance(i));
                ListInstancePerLabel.put(labelSet, li);
            }
        }

        // Iterates the structure and a) if occurences of a labelset are higher
        // than p parameter then add them to the training set, b) if occurences
        // are less, then depending on the strategy discard/reintroduce them
        Instances newData = new Instances(data, 0);
        Iterator<LabelSet> it = ListInstancePerLabel.keySet().iterator();
        while (it.hasNext()) {
            LabelSet ls = it.next();
            ArrayList<Instance> instances = ListInstancePerLabel.get(ls);
            if (instances.size() > p) {
                for (int i = 0; i < instances.size(); i++) {
                    newData.add(instances.get(i));
                }
            } else {
                ArrayList<Instance> processed = processRejected(ls);
                newData.addAll(processed);
            }
        }

        super.buildInternal(new MultiLabelInstances(newData, mlDataSet.getLabelsMetaData()));
    }
}