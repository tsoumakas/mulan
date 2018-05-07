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

import mulan.classifier.MultiLabelLearnerBase;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

/**
 * Base class for multi-label learners that are based on data transformation
 *
 * @author Robert Friberg
 * @author Jozef Vilcek
 * @author Grigorios Tsoumakas
 * @version 2010.12.25
 */
@SuppressWarnings("serial")
public abstract class TransformationBasedMultiLabelLearner extends MultiLabelLearnerBase {

    /**
     * The underlying single-label classifier.
     */
    protected Classifier baseClassifier;

    /**
     * Creates a new instance of {@link TransformationBasedMultiLabelLearner}
     * with default {@link J48} base classifier.
     */
    public TransformationBasedMultiLabelLearner() {
        this(new J48());
    }

    /**
     * Creates a new instance.
     *
     * @param baseClassifier the base classifier which will be used internally
     * to handle the data.
     * @see Classifier
     */
    public TransformationBasedMultiLabelLearner(Classifier baseClassifier) {
        // todo: check if it is not a regressor
        this.baseClassifier = baseClassifier;
    }

    /**
     * Returns the {@link Classifier} which is used internally by the learner.
     *
     * @return the internally used classifier
     */
    public Classifier getBaseClassifier() {
        return baseClassifier;
    }

    /**
     * Returns an instance of a TechnicalInformation object, containing detailed
     * information about the technical background of this class, e.g., paper
     * reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(Type.INCOLLECTION);
        result.setValue(Field.AUTHOR, "Tsoumakas, Grigorios and Katakis, Ioannis and Vlahavas, Ioannis");
        result.setValue(Field.TITLE, "Mining Multi-Label Data");
        result.setValue(Field.PAGES, "667-685");
        result.setValue(Field.BOOKTITLE, "Data Mining and Knowledge Discovery Handbook");
        result.setValue(Field.EDITOR, "Maimon, Oded and Rokach, Lior");
        result.setValue(Field.PUBLISHER, "Springer");
        result.setValue(Field.EDITION, "2nd");
        result.setValue(Field.YEAR, "2010");
        return result;
    }

    /**
     * Returns a string describing the classifier.
     *
     * @return a string description of the classifier
     */
    public String globalInfo() {
        return "Base class for multi-label learners, which use problem "
                + "transformation to handle multi-label data. "
                + "For more information, see\n\n"
                + getTechnicalInformation().toString();
    }
}