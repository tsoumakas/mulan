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
 *    LabelsBuilderException.java
 *    Copyright (C) 2009-2012 Aristotle University of Thessaloniki, Greece
 */
package mulan.data;

import mulan.core.MulanException;

/**
 * Exception is raised by {@link LabelsBuilder} to indicate an error when creating 
 * {@link LabelsMetaDataImpl} instance form specified source.
 * 
 * @author Jozef Vilcek
 * @see LabelsBuilder
 */
public class LabelsBuilderException extends MulanException {

    private static final long serialVersionUID = 2161709838882541792L;

    /**
     * Creates a new {@link LabelsBuilderException} instance.
     * @param message the detail message
     */
    public LabelsBuilderException(String message) {
        super(message);
    }

    /**
     * Creates a new instance of {@link LabelsBuilderException} with the specified
     * detail message and nested exception.
     *
     * @param message the detail message
     * @param cause the nested exception
     */
    public LabelsBuilderException(String message, Throwable cause) {
        super(message, cause);
    }
}