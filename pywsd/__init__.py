#!/usr/bin/env python -*- coding: utf-8 -*-
#
# Python Word Sense Disambiguation (pyWSD)
#
# Copyright (C) 2014-2017 alvations
# URL:
# For license information, see LICENSE.md

from __future__ import absolute_import

from .lesk import *
from .baseline import *
from .similarity import *

#import semcor
#import semeval

from .allwords_wsd import disambiguate
