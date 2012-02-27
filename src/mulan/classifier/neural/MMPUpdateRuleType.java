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
 *    MMPUpdateRuleType.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.classifier.neural;

/**
 * The enumeration of update rules, which can be used by the 
 * {@link MMPLearner} to update its model in learning phase.
 * 
 * @author Jozef Vilcek
 * @version 2012.02.27
 */
public enum MMPUpdateRuleType {

    /** Indicates usage of uniform update rule */
    UniformUpdate,
    /** Indicates usage of maximum update rule */
    MaxUpdate,
    /** Indicates usage of randomized update rule */
    RandomizedUpdate,
}