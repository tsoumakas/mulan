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
 *    MultiLabelMetaLearner.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.classifier.meta;

import mulan.classifier.*;
import mulan.core.ArgumentNullException;

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
}