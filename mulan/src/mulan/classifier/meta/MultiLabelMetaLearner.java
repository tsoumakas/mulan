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
package mulan.classifier.meta;

import mulan.classifier.*;
import mulan.core.ArgumentNullException;
import weka.core.TechnicalInformation;

/**
 * Base class for multi-label learners, which use other multi-label learners
 *
 * @author Grigorios Tsoumakas
 * @version $Revision: 0.01 $
 */
public abstract class MultiLabelMetaLearner extends MultiLabelLearnerBase {

    /**
     * The encapsulated classifier or used for making clones in the case of
     * ensemble classifiers.
     */
    protected final MultiLabelLearner baseLearner;

    /**
     * Creates a new instance.
     *
     * @param baseLearner the base multi-label learner which will be used
     * internally to handle the data.
     */
    public MultiLabelMetaLearner(MultiLabelLearner baseLearner) {
        if (baseLearner == null) {
            throw new ArgumentNullException("baseLearner");
        }
        this.baseLearner = baseLearner;
    }

    /**
     * Returns the {@link MultiLabelLearner} which is used internally by the
     * learner.
     *
     * @return the baseLearner
     */
    public MultiLabelLearner getBaseLearner() {
        return baseLearner;
    }

    /**
     * Returns an instance of a TechnicalInformation object, containing detailed
     * information about the technical background of this class, e.g., paper
     * reference or book this class is based on.
     *
     * @return the technical information about this class
     */
    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result = new TechnicalInformation(TechnicalInformation.Type.INCOLLECTION);
        result.setValue(TechnicalInformation.Field.AUTHOR, "Tsoumakas, Grigorios and Katakis, Ioannis and Vlahavas, Ioannis");
        result.setValue(TechnicalInformation.Field.TITLE, "Mining Multi-Label Data");
        result.setValue(TechnicalInformation.Field.PAGES, "667-685");
        result.setValue(TechnicalInformation.Field.BOOKTITLE, "Data Mining and Knowledge Discovery Handbook");
        result.setValue(TechnicalInformation.Field.EDITOR, "Maimon, Oded and Rokach, Lior");
        result.setValue(TechnicalInformation.Field.PUBLISHER, "Springer");
        result.setValue(TechnicalInformation.Field.EDITION, "2nd");
        result.setValue(TechnicalInformation.Field.YEAR, "2010");
        return result;
    }
}