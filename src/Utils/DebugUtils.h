/*
 * PTracking - Distributed real-time multiple object tracking library.
 * Copyright (c) 2014, Fabio Previtali. All rights reserved.
 * 
 * This file is part of PTracking.
 * 
 * PTracking is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * PTracking is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with PTracking. If not, see <http://www.gnu.org/licenses/>.
 * 
 * Written by Fabio Previtali.
 * 
 * Please, report any suggestion/comment/bug to fabio.previtali@gmail.com.
 */

#pragma once

#include <iostream>

#define DEBUG(x) std::cerr << "\033[22;34;1m" << x << "\033[0m";
#define ERR(x) std::cerr << "\033[22;31;1m" << x << "\033[0m";
#define INFO(x) std::cerr << "\033[22;37;1m" << x << "\033[0m";
#define LOG(x) std::cerr << "\033[22;38;1m" << x << "\033[0m";
#define WARN(x) std::cerr << "\033[22;33;1m" << x << "\033[0m";
