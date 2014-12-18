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
 *    NormalizationFilter.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.classifier.neural;

import java.io.Serializable;
import java.util.Hashtable;
import java.util.Set;
import mulan.data.MultiLabelInstances;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.experiment.Stats;


/**
 * Performs a normalization of numeric attributes of the data set.
 * It is initialized based on given {@link MultiLabelInstances} data set and then can 
 * be used to normalize {@link Instance} instances which conform to the format of the 
 * data set the {@link NormalizationFilter} was initialized from. 
 * 
 * @author Jozef Vilcek
 * @version 2012.02.27
 */
public class NormalizationFilter implements Serializable {

    /** Serial UID for serialization */
    private static final long serialVersionUID = -2575012048861337275L;
    private final double maxValue;
    private final double minValue;
    private Hashtable<Integer, double[]> attStats;

    /**
     * Creates a new instance of {@link NormalizationFilter} class for given data set.
     *
     * @param mlData the {@link MultiLabelInstances} data set from which normalizer
     * should be initialized.
     * @param performNormalization indicates whether normalization of instances contained
     * in the data set used for initialization should be performed
     * @param minValue the minimum value of the normalization range for numerical attributes
     * @param maxValue the maximum value of the normalization range for numerical attributes
     */
    public NormalizationFilter(MultiLabelInstances mlData, boolean performNormalization, double minValue, double maxValue) {
        if (mlData == null) {
            throw new IllegalArgumentException("Parameter 'mlData' is null.");
        }
        if (maxValue <= minValue) {
            throw new IllegalArgumentException(
                    String.format("Parameters 'minValue=%f' and 'maxValue=%f' does not define valid range.",
                    minValue, maxValue));
        }

        this.minValue = minValue;
        this.maxValue = maxValue;
        this.attStats = new Hashtable<Integer, double[]>();

        Initialize(mlData);

        if (performNormalization) {
            Instances instances = mlData.getDataSet();
            int numInstance = instances.numInstances();
            for (int i = 0; i < numInstance; i++) {
                normalize(instances.instance(i));
            }
        }
    }

    /**
     * Creates a new instance of {@link NormalizationFilter} class for given data set.
     * The normalizer will be initialized to perform normalization to the default
     * range &lt;-1,1&gt;.
     *
     * @param mlData the {@link MultiLabelInstances} data set from which normalizer
     * should be initialized.
     * @param performNormalization indicates whether normalization of instances contained
     * in the data set used for initialization should be performed
     */
    public NormalizationFilter(MultiLabelInstances mlData, boolean performNormalization) {
        this(mlData, performNormalization, -1, 1);
    }

    /**
     * Performs a normalization of numerical attributes on given instance.
     * The instance must conform to format of instances data the {@link NormalizationFilter}
     * was initialized with.
     * @param instance the instance to be normalized
     */
    public void normalize(Instance instance) {
        Set<Integer> normScope = attStats.keySet();
        for (Integer attIndex : normScope) {
            double[] stats = attStats.get(attIndex);
            double attMin = stats[0];
            double attMax = stats[1];
            double value = instance.value(attIndex);
            if (attMin == attMax) {
                instance.setValue(attIndex, minValue);
            } else {
                instance.setValue(attIndex, (((value - stats[0]) / (stats[1] - stats[0])) * (maxValue - minValue)) + minValue);
            }
        }
    }

    private void Initialize(MultiLabelInstances mlData) {
        Instances dataSet = mlData.getDataSet();
        int[] featureIndices = mlData.getFeatureIndices();
        for (int attIndex : featureIndices) {
            Attribute feature = dataSet.attribute(attIndex);
            if (feature.isNumeric()) {
                Stats stats = dataSet.attributeStats(attIndex).numericStats;
                attStats.put(attIndex, new double[]{stats.min, stats.max});
            }
        }
    }
}