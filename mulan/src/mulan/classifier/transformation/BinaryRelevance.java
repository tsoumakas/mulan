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
 *    BinaryRelevance.java
 *    Copyright (C) 2009-2010 Aristotle University of Thessaloniki, Thessaloniki, Greece
 */
package mulan.classifier.transformation;

import mulan.classifier.MultiLabelOutput;
import mulan.data.MultiLabelInstances;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;

/**
 * 
 * <!-- globalinfo-start -->
 * 
 * Class that implements the Binary Relevance (BR) method. For more information,
 * see <br/>
 * <br/>
 * G. Tsoumakas, I. Katakis, I. Vlahavas, "Mining Multi-label Data", Data Mining
 * and Knowledge Discovery Handbook (draft of preliminary accepted chapter), O.
 * Maimon, L. Rokach (Ed.), 2nd edition, Springer, 2009. </p>
 * 
 * <!-- globalinfo-end -->
 * 
 * <!-- technical-bibtex-start --> BibTeX:
 * 
 * <pre>
 * &#064;inbook{tsoumakas+etal:2009,
 *    author =    {Tsoumakas, G. and Katakis, I. and Vlahavas, I.},
 *    title =     {Mining Multi-label Data},
 *    booktitle = {Data Mining and Knowledge Discovery Handbook, 2nd edition},
 *    year =      {2009},
 *    editor =    {Maimon, O. and Rokach, L.},
 * }
 * </pre>
 * 
 * <p/>
 * <!-- technical-bibtex-end -->
 * 
 * @author Robert Friberg
 * @author Grigorios Tsoumakas
 * @version $Revision: 0.06$
 */
public class BinaryRelevance extends TransformationBasedMultiLabelLearner {

    /**
     * The ensemble of binary relevance models. These are Weka FilteredClassifier
     * objects, where the filter corresponds to removing all label apart from
     * the one that serves as a target for the corresponding model.
     */
    protected FilteredClassifier[] ensemble;

    /**
     * Creates a new instance
     *
     * @param classifier the base-level classification algorithm that will
     * be used for training each of the binary models
     */
    public BinaryRelevance(Classifier classifier) {
        super(classifier);
    }

	/**
	 * The correspondence between ensemble models and labels
	 */
	private String[] correspondence;

	protected void buildInternal(MultiLabelInstances train) throws Exception {
		numLabels = train.getNumLabels();
		ensemble = new FilteredClassifier[numLabels];
		correspondence = new String[numLabels];
		Instances trainingData = train.getDataSet();
		for (int i = 0; i < numLabels; i++) {
			ensemble[i] = new FilteredClassifier();
			ensemble[i].setClassifier(AbstractClassifier.makeCopy(baseClassifier));

			// Indices of attributes to remove
			int[] indicesToRemove = new int[numLabels - 1];
			int counter2 = 0;
			for (int counter1 = 0; counter1 < numLabels; counter1++) {
				if (labelIndices[counter1] != labelIndices[i]) {
					indicesToRemove[counter2] = labelIndices[counter1];
					counter2++;
				}
			}

			Remove remove = new Remove();
			remove.setAttributeIndicesArray(indicesToRemove);
			remove.setInputFormat(trainingData);
			remove.setInvertSelection(false);
			ensemble[i].setFilter(remove);

			trainingData.setClassIndex(labelIndices[i]);
			correspondence[i] = trainingData.classAttribute().name();
			debug("Bulding model " + (i + 1) + "/" + numLabels);
			ensemble[i].buildClassifier(trainingData);
		}
	}

    protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception {
        boolean[] bipartition = new boolean[numLabels];
        double[] confidences = new double[numLabels];

		for (int counter = 0; counter < numLabels; counter++) {
			double distribution[] = new double[2];
			try {
				distribution = ensemble[counter].distributionForInstance(instance);
			} catch (Exception e) {
				System.out.println(e);
				return null;
			}
			int maxIndex = (distribution[0] > distribution[1]) ? 0 : 1;

			// Ensure correct predictions both for class values {0,1} and {1,0}
			Attribute classAttribute = ensemble[counter].getFilter().getOutputFormat().classAttribute();
			bipartition[counter] = (classAttribute.value(maxIndex).equals("1")) ? true : false;

			// The confidence of the label being equal to 1
			confidences[counter] = distribution[classAttribute.indexOfValue("1")];
		}

		MultiLabelOutput mlo = new MultiLabelOutput(bipartition, confidences);
		return mlo;
	}

	/**
	 * Returns the model which corresponds to the label with labelName
	 * 
	 * @param labelName
	 * @return the corresponding model or null if the labelIndex is wrong
	 */
	public Classifier getModel(String labelName) {
		for (int i = 0; i < numLabels; i++) {
			if (correspondence[i].equals(labelName)) {
				return ensemble[i].getClassifier();
			}
		}
		return null;
	}
}
