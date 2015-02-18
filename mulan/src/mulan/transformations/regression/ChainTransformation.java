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
package mulan.transformations.regression;

import java.io.Serializable;

import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

/**
 * This class implements the Classifier/Regressor Chain transformation.
 * 
 * @author Eleftherios Spyromitros-Xioufis
 * @version 2014.04.01
 */
public class ChainTransformation implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * Deletes all target attributes that appear after the first targetsToKeep in the chain. The
     * target attribute at position targetsToKeep in the chain is set as the class attribute.
     * 
     * @param data the input data set
     * @param chain a chain (permutation) of the indices of the target attributes
     * @param numTargetsToKeep the number of target attributes from the beginning of the chain that
     *            should be kept, 1&lt;=numTargetsToKeep&lt;=numOfTargets
     * @return the transformed Instances object. The input object is not modified.
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    public static Instances transformInstances(Instances data, int[] chain, int numTargetsToKeep)
            throws Exception {
        int numOfTargets = chain.length;
        if (numTargetsToKeep < 1 || numTargetsToKeep > numOfTargets) {
            throw new Exception("keepFirstKTargets should be between 1 and numOfTargets");
        }
        // Indices of attributes to remove
        int[] indicesToRemove = new int[numOfTargets - numTargetsToKeep];
        // the indices of the target attributes whose position in the chain is
        // after the first keepFirstKTargets attributes are marked for removal
        for (int i = 0; i < numOfTargets - numTargetsToKeep; i++) {
            indicesToRemove[i] = chain[numTargetsToKeep + i];
        }

        Remove remove = new Remove();
        remove.setAttributeIndicesArray(indicesToRemove);
        remove.setInputFormat(data);
        // get the class attribute name, the name of the target attribute which is placed in the
        // targetsToKeep position of the chain
        String classAttributeName = data.attribute(chain[numTargetsToKeep - 1]).name();
        Instances transformed = Filter.useFilter(data, remove);
        transformed.setClass(transformed.attribute(classAttributeName));
        return transformed;
    }

    /**
     * Transforms a single Instance in the same way as
     * {@link #transformInstance(Instance, int[], int)} transforms an Instances object.
     * 
     * @param instance the input instance
     * @param chain a chain (permutation) of the indices of the target attributes
     * @param numTargetsToKeep the number of target attributes from the beginning of the chain that
     *            should be kept, 1&lt;=numTargetsToKeep&lt;=numOfTargets
     * @return the transformed Instance object. The input object is not modified.
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    public static Instance transformInstance(Instance instance, int[] chain, int numTargetsToKeep)
            throws Exception {
        int numOfTargets = chain.length;
        // Indices of attributes to remove
        int[] indicesToRemove = new int[numOfTargets - numTargetsToKeep];
        for (int i = 0; i < numOfTargets - numTargetsToKeep; i++) {
            indicesToRemove[i] = chain[numTargetsToKeep + i];
        }
        Remove remove = new Remove();
        remove.setAttributeIndicesArray(indicesToRemove);
        remove.setInputFormat(instance.dataset());
        remove.input(instance);
        remove.batchFinished();
        Instance transformed = remove.output();

        return transformed;
    }

    /**
     * Exemplifies how the per instance transformation works.
     * 
     * @param args Arguments accepted from command line
     * @throws Exception Potential exception thrown. To be handled in an upper level.
     */
    public static void main(String[] args) throws Exception {

        String trainpath = "data/solar-flare_1.arff";
        int numTargets = 3;
        MultiLabelInstances train = new MultiLabelInstances(trainpath, numTargets);
        int[] targetIndices = train.getLabelIndices();

        Instance original = train.getDataSet().instance(0);
        System.out.println("Original:\t" + original);
        for (int i = 1; i <= train.getNumLabels(); i++) {
            Instance transformed = ChainTransformation
                    .transformInstance(original, targetIndices, i);
            System.out.println("Transformed " + i + ":\t" + transformed);
        }
    }
}
