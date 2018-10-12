# -*- encoding: utf-8 -*-

import numpy as np
import nnhealpix as nnh

def test_dgrade():
    ref41 = np.array([
        0, 4, 5, 12, 13, 14, 24, 25, 26, 27, 41, 42, 43, 57, 58,
        74, 1, 6, 7, 15, 16, 17, 28, 29, 30, 31, 45, 46, 47, 61,
        62, 78, 2, 8, 9, 18, 19, 20, 32, 33, 34, 35, 49, 50, 51,
        65, 66, 82, 3, 10, 11, 21, 22, 23, 36, 37, 38, 39, 53, 54,
        55, 69, 70, 86, 40, 56, 71, 72, 73, 87, 88, 89, 102, 103, 104,
        105, 119, 120, 135, 136, 44, 59, 60, 75, 76, 77, 90, 91, 92, 93,
        107, 108, 109, 123, 124, 140, 48, 63, 64, 79, 80, 81, 94, 95, 96,
        97, 111, 112, 113, 127, 128, 144, 52, 67, 68, 83, 84, 85, 98, 99,
        100, 101, 115, 116, 117, 131, 132, 148, 106, 121, 122, 137, 138, 139, 152,
        153, 154, 155, 168, 169, 170, 180, 181, 188, 110, 125, 126, 141, 142, 143,
        156, 157, 158, 159, 171, 172, 173, 182, 183, 189, 114, 129, 130, 145, 146,
        147, 160, 161, 162, 163, 174, 175, 176, 184, 185, 190, 118, 133, 134, 149,
        150, 151, 164, 165, 166, 167, 177, 178, 179, 186, 187, 191,
    ], dtype='int')

    ref42 = np.array([
        0, 4, 5, 13, 1, 6, 7, 16, 2, 8, 9, 19, 3, 10, 11,
        22, 12, 24, 25, 41, 14, 26, 27, 43, 15, 28, 29, 45, 17, 30,
        31, 47, 18, 32, 33, 49, 20, 34, 35, 51, 21, 36, 37, 53, 23,
        38, 39, 55, 40, 56, 71, 72, 42, 57, 58, 74, 44, 59, 60, 76,
        46, 61, 62, 78, 48, 63, 64, 80, 50, 65, 66, 82, 52, 67, 68,
        84, 54, 69, 70, 86, 73, 88, 89, 105, 75, 90, 91, 107, 77, 92,
        93, 109, 79, 94, 95, 111, 81, 96, 97, 113, 83, 98, 99, 115, 85,
        100, 101, 117, 87, 102, 103, 119, 104, 120, 135, 136, 106, 121, 122, 138,
        108, 123, 124, 140, 110, 125, 126, 142, 112, 127, 128, 144, 114, 129, 130,
        146, 116, 131, 132, 148, 118, 133, 134, 150, 137, 152, 153, 168, 139, 154,
        155, 170, 141, 156, 157, 171, 143, 158, 159, 173, 145, 160, 161, 174, 147,
        162, 163, 176, 149, 164, 165, 177, 151, 166, 167, 179, 169, 180, 181, 188,
        172, 182, 183, 189, 175, 184, 185, 190, 178, 186, 187, 191,
    ], dtype='int')

    ref81 = np.array([
        0, 4, 5, 12, 13, 14, 24, 25, 26, 27, 40, 41, 42, 43, 44,
        60, 61, 62, 63, 64, 65, 84, 85, 86, 87, 88, 89, 90, 112, 113,
        114, 115, 116, 117, 118, 119, 145, 146, 147, 148, 149, 150, 151, 177, 178,
        179, 180, 181, 182, 210, 211, 212, 213, 214, 242, 243, 244, 245, 275, 276,
        277, 307, 308, 340, 1, 6, 7, 15, 16, 17, 28, 29, 30, 31, 45,
        46, 47, 48, 49, 66, 67, 68, 69, 70, 71, 91, 92, 93, 94, 95,
        96, 97, 120, 121, 122, 123, 124, 125, 126, 127, 153, 154, 155, 156, 157,
        158, 159, 185, 186, 187, 188, 189, 190, 218, 219, 220, 221, 222, 250, 251,
        252, 253, 283, 284, 285, 315, 316, 348, 2, 8, 9, 18, 19, 20, 32,
        33, 34, 35, 50, 51, 52, 53, 54, 72, 73, 74, 75, 76, 77, 98,
        99, 100, 101, 102, 103, 104, 128, 129, 130, 131, 132, 133, 134, 135, 161,
        162, 163, 164, 165, 166, 167, 193, 194, 195, 196, 197, 198, 226, 227, 228,
        229, 230, 258, 259, 260, 261, 291, 292, 293, 323, 324, 356, 3, 10, 11,
        21, 22, 23, 36, 37, 38, 39, 55, 56, 57, 58, 59, 78, 79, 80,
        81, 82, 83, 105, 106, 107, 108, 109, 110, 111, 136, 137, 138, 139, 140,
        141, 142, 143, 169, 170, 171, 172, 173, 174, 175, 201, 202, 203, 204, 205,
        206, 234, 235, 236, 237, 238, 266, 267, 268, 269, 299, 300, 301, 331, 332,
        364, 144, 176, 207, 208, 209, 239, 240, 241, 270, 271, 272, 273, 274, 302,
        303, 304, 305, 306, 333, 334, 335, 336, 337, 338, 339, 365, 366, 367, 368,
        369, 370, 371, 396, 397, 398, 399, 400, 401, 402, 403, 429, 430, 431, 432,
        433, 434, 461, 462, 463, 464, 465, 466, 494, 495, 496, 497, 526, 527, 528,
        529, 559, 560, 591, 592, 152, 183, 184, 215, 216, 217, 246, 247, 248, 249,
        278, 279, 280, 281, 282, 309, 310, 311, 312, 313, 314, 341, 342, 343, 344,
        345, 346, 347, 372, 373, 374, 375, 376, 377, 378, 379, 405, 406, 407, 408,
        409, 410, 411, 437, 438, 439, 440, 441, 442, 470, 471, 472, 473, 474, 502,
        503, 504, 505, 535, 536, 537, 567, 568, 600, 160, 191, 192, 223, 224, 225,
        254, 255, 256, 257, 286, 287, 288, 289, 290, 317, 318, 319, 320, 321, 322,
        349, 350, 351, 352, 353, 354, 355, 380, 381, 382, 383, 384, 385, 386, 387,
        413, 414, 415, 416, 417, 418, 419, 445, 446, 447, 448, 449, 450, 478, 479,
        480, 481, 482, 510, 511, 512, 513, 543, 544, 545, 575, 576, 608, 168, 199,
        200, 231, 232, 233, 262, 263, 264, 265, 294, 295, 296, 297, 298, 325, 326,
        327, 328, 329, 330, 357, 358, 359, 360, 361, 362, 363, 388, 389, 390, 391,
        392, 393, 394, 395, 421, 422, 423, 424, 425, 426, 427, 453, 454, 455, 456,
        457, 458, 486, 487, 488, 489, 490, 518, 519, 520, 521, 551, 552, 553, 583,
        584, 616, 404, 435, 436, 467, 468, 469, 498, 499, 500, 501, 530, 531, 532,
        533, 534, 561, 562, 563, 564, 565, 566, 593, 594, 595, 596, 597, 598, 599,
        624, 625, 626, 627, 628, 629, 630, 631, 656, 657, 658, 659, 660, 661, 662,
        684, 685, 686, 687, 688, 689, 708, 709, 710, 711, 712, 728, 729, 730, 731,
        744, 745, 746, 756, 757, 764, 412, 443, 444, 475, 476, 477, 506, 507, 508,
        509, 538, 539, 540, 541, 542, 569, 570, 571, 572, 573, 574, 601, 602, 603,
        604, 605, 606, 607, 632, 633, 634, 635, 636, 637, 638, 639, 663, 664, 665,
        666, 667, 668, 669, 690, 691, 692, 693, 694, 695, 713, 714, 715, 716, 717,
        732, 733, 734, 735, 747, 748, 749, 758, 759, 765, 420, 451, 452, 483, 484,
        485, 514, 515, 516, 517, 546, 547, 548, 549, 550, 577, 578, 579, 580, 581,
        582, 609, 610, 611, 612, 613, 614, 615, 640, 641, 642, 643, 644, 645, 646,
        647, 670, 671, 672, 673, 674, 675, 676, 696, 697, 698, 699, 700, 701, 718,
        719, 720, 721, 722, 736, 737, 738, 739, 750, 751, 752, 760, 761, 766, 428,
        459, 460, 491, 492, 493, 522, 523, 524, 525, 554, 555, 556, 557, 558, 585,
        586, 587, 588, 589, 590, 617, 618, 619, 620, 621, 622, 623, 648, 649, 650,
        651, 652, 653, 654, 655, 677, 678, 679, 680, 681, 682, 683, 702, 703, 704,
        705, 706, 707, 723, 724, 725, 726, 727, 740, 741, 742, 743, 753, 754, 755,
        762, 763, 767,
    ], dtype='int')
    
    ref82 = np.array([
        0, 4, 5, 12, 13, 14, 24, 25, 26, 27, 41, 42, 43, 62, 63,
        87, 1, 6, 7, 15, 16, 17, 28, 29, 30, 31, 46, 47, 48, 68,
        69, 94, 2, 8, 9, 18, 19, 20, 32, 33, 34, 35, 51, 52, 53,
        74, 75, 101, 3, 10, 11, 21, 22, 23, 36, 37, 38, 39, 56, 57,
        58, 80, 81, 108, 40, 60, 61, 84, 85, 86, 112, 113, 114, 115, 145,
        146, 147, 177, 178, 210, 44, 64, 65, 88, 89, 90, 116, 117, 118, 119,
        149, 150, 151, 181, 182, 214, 45, 66, 67, 91, 92, 93, 120, 121, 122,
        123, 153, 154, 155, 185, 186, 218, 49, 70, 71, 95, 96, 97, 124, 125,
        126, 127, 157, 158, 159, 189, 190, 222, 50, 72, 73, 98, 99, 100, 128,
        129, 130, 131, 161, 162, 163, 193, 194, 226, 54, 76, 77, 102, 103, 104,
        132, 133, 134, 135, 165, 166, 167, 197, 198, 230, 55, 78, 79, 105, 106,
        107, 136, 137, 138, 139, 169, 170, 171, 201, 202, 234, 59, 82, 83, 109,
        110, 111, 140, 141, 142, 143, 173, 174, 175, 205, 206, 238, 144, 176, 207,
        208, 209, 239, 240, 241, 270, 271, 272, 273, 303, 304, 335, 336, 148, 179,
        180, 211, 212, 213, 242, 243, 244, 245, 275, 276, 277, 307, 308, 340, 152,
        183, 184, 215, 216, 217, 246, 247, 248, 249, 279, 280, 281, 311, 312, 344,
        156, 187, 188, 219, 220, 221, 250, 251, 252, 253, 283, 284, 285, 315, 316,
        348, 160, 191, 192, 223, 224, 225, 254, 255, 256, 257, 287, 288, 289, 319,
        320, 352, 164, 195, 196, 227, 228, 229, 258, 259, 260, 261, 291, 292, 293,
        323, 324, 356, 168, 199, 200, 231, 232, 233, 262, 263, 264, 265, 295, 296,
        297, 327, 328, 360, 172, 203, 204, 235, 236, 237, 266, 267, 268, 269, 299,
        300, 301, 331, 332, 364, 274, 305, 306, 337, 338, 339, 368, 369, 370, 371,
        401, 402, 403, 433, 434, 466, 278, 309, 310, 341, 342, 343, 372, 373, 374,
        375, 405, 406, 407, 437, 438, 470, 282, 313, 314, 345, 346, 347, 376, 377,
        378, 379, 409, 410, 411, 441, 442, 474, 286, 317, 318, 349, 350, 351, 380,
        381, 382, 383, 413, 414, 415, 445, 446, 478, 290, 321, 322, 353, 354, 355,
        384, 385, 386, 387, 417, 418, 419, 449, 450, 482, 294, 325, 326, 357, 358,
        359, 388, 389, 390, 391, 421, 422, 423, 453, 454, 486, 298, 329, 330, 361,
        362, 363, 392, 393, 394, 395, 425, 426, 427, 457, 458, 490, 302, 333, 334,
        365, 366, 367, 396, 397, 398, 399, 429, 430, 431, 461, 462, 494, 400, 432,
        463, 464, 465, 495, 496, 497, 526, 527, 528, 529, 559, 560, 591, 592, 404,
        435, 436, 467, 468, 469, 498, 499, 500, 501, 531, 532, 533, 563, 564, 596,
        408, 439, 440, 471, 472, 473, 502, 503, 504, 505, 535, 536, 537, 567, 568,
        600, 412, 443, 444, 475, 476, 477, 506, 507, 508, 509, 539, 540, 541, 571,
        572, 604, 416, 447, 448, 479, 480, 481, 510, 511, 512, 513, 543, 544, 545,
        575, 576, 608, 420, 451, 452, 483, 484, 485, 514, 515, 516, 517, 547, 548,
        549, 579, 580, 612, 424, 455, 456, 487, 488, 489, 518, 519, 520, 521, 551,
        552, 553, 583, 584, 616, 428, 459, 460, 491, 492, 493, 522, 523, 524, 525,
        555, 556, 557, 587, 588, 620, 530, 561, 562, 593, 594, 595, 624, 625, 626,
        627, 656, 657, 658, 684, 685, 708, 534, 565, 566, 597, 598, 599, 628, 629,
        630, 631, 660, 661, 662, 688, 689, 712, 538, 569, 570, 601, 602, 603, 632,
        633, 634, 635, 663, 664, 665, 690, 691, 713, 542, 573, 574, 605, 606, 607,
        636, 637, 638, 639, 667, 668, 669, 694, 695, 717, 546, 577, 578, 609, 610,
        611, 640, 641, 642, 643, 670, 671, 672, 696, 697, 718, 550, 581, 582, 613,
        614, 615, 644, 645, 646, 647, 674, 675, 676, 700, 701, 722, 554, 585, 586,
        617, 618, 619, 648, 649, 650, 651, 677, 678, 679, 702, 703, 723, 558, 589,
        590, 621, 622, 623, 652, 653, 654, 655, 681, 682, 683, 706, 707, 727, 659,
        686, 687, 709, 710, 711, 728, 729, 730, 731, 744, 745, 746, 756, 757, 764,
        666, 692, 693, 714, 715, 716, 732, 733, 734, 735, 747, 748, 749, 758, 759,
        765, 673, 698, 699, 719, 720, 721, 736, 737, 738, 739, 750, 751, 752, 760,
        761, 766, 680, 704, 705, 724, 725, 726, 740, 741, 742, 743, 753, 754, 755,
        762, 763, 767,
    ], dtype='int')
    
    ref84 = np.array([
        0, 4, 5, 13, 1, 6, 7, 16, 2, 8, 9, 19, 3, 10, 11,
        22, 12, 24, 25, 41, 14, 26, 27, 43, 15, 28, 29, 46, 17, 30,
        31, 48, 18, 32, 33, 51, 20, 34, 35, 53, 21, 36, 37, 56, 23,
        38, 39, 58, 40, 60, 61, 85, 42, 62, 63, 87, 44, 64, 65, 89,
        45, 66, 67, 92, 47, 68, 69, 94, 49, 70, 71, 96, 50, 72, 73,
        99, 52, 74, 75, 101, 54, 76, 77, 103, 55, 78, 79, 106, 57, 80,
        81, 108, 59, 82, 83, 110, 84, 112, 113, 145, 86, 114, 115, 147, 88,
        116, 117, 149, 90, 118, 119, 151, 91, 120, 121, 153, 93, 122, 123, 155,
        95, 124, 125, 157, 97, 126, 127, 159, 98, 128, 129, 161, 100, 130, 131,
        163, 102, 132, 133, 165, 104, 134, 135, 167, 105, 136, 137, 169, 107, 138,
        139, 171, 109, 140, 141, 173, 111, 142, 143, 175, 144, 176, 207, 208, 146,
        177, 178, 210, 148, 179, 180, 212, 150, 181, 182, 214, 152, 183, 184, 216,
        154, 185, 186, 218, 156, 187, 188, 220, 158, 189, 190, 222, 160, 191, 192,
        224, 162, 193, 194, 226, 164, 195, 196, 228, 166, 197, 198, 230, 168, 199,
        200, 232, 170, 201, 202, 234, 172, 203, 204, 236, 174, 205, 206, 238, 209,
        240, 241, 273, 211, 242, 243, 275, 213, 244, 245, 277, 215, 246, 247, 279,
        217, 248, 249, 281, 219, 250, 251, 283, 221, 252, 253, 285, 223, 254, 255,
        287, 225, 256, 257, 289, 227, 258, 259, 291, 229, 260, 261, 293, 231, 262,
        263, 295, 233, 264, 265, 297, 235, 266, 267, 299, 237, 268, 269, 301, 239,
        270, 271, 303, 272, 304, 335, 336, 274, 305, 306, 338, 276, 307, 308, 340,
        278, 309, 310, 342, 280, 311, 312, 344, 282, 313, 314, 346, 284, 315, 316,
        348, 286, 317, 318, 350, 288, 319, 320, 352, 290, 321, 322, 354, 292, 323,
        324, 356, 294, 325, 326, 358, 296, 327, 328, 360, 298, 329, 330, 362, 300,
        331, 332, 364, 302, 333, 334, 366, 337, 368, 369, 401, 339, 370, 371, 403,
        341, 372, 373, 405, 343, 374, 375, 407, 345, 376, 377, 409, 347, 378, 379,
        411, 349, 380, 381, 413, 351, 382, 383, 415, 353, 384, 385, 417, 355, 386,
        387, 419, 357, 388, 389, 421, 359, 390, 391, 423, 361, 392, 393, 425, 363,
        394, 395, 427, 365, 396, 397, 429, 367, 398, 399, 431, 400, 432, 463, 464,
        402, 433, 434, 466, 404, 435, 436, 468, 406, 437, 438, 470, 408, 439, 440,
        472, 410, 441, 442, 474, 412, 443, 444, 476, 414, 445, 446, 478, 416, 447,
        448, 480, 418, 449, 450, 482, 420, 451, 452, 484, 422, 453, 454, 486, 424,
        455, 456, 488, 426, 457, 458, 490, 428, 459, 460, 492, 430, 461, 462, 494,
        465, 496, 497, 529, 467, 498, 499, 531, 469, 500, 501, 533, 471, 502, 503,
        535, 473, 504, 505, 537, 475, 506, 507, 539, 477, 508, 509, 541, 479, 510,
        511, 543, 481, 512, 513, 545, 483, 514, 515, 547, 485, 516, 517, 549, 487,
        518, 519, 551, 489, 520, 521, 553, 491, 522, 523, 555, 493, 524, 525, 557,
        495, 526, 527, 559, 528, 560, 591, 592, 530, 561, 562, 594, 532, 563, 564,
        596, 534, 565, 566, 598, 536, 567, 568, 600, 538, 569, 570, 602, 540, 571,
        572, 604, 542, 573, 574, 606, 544, 575, 576, 608, 546, 577, 578, 610, 548,
        579, 580, 612, 550, 581, 582, 614, 552, 583, 584, 616, 554, 585, 586, 618,
        556, 587, 588, 620, 558, 589, 590, 622, 593, 624, 625, 656, 595, 626, 627,
        658, 597, 628, 629, 660, 599, 630, 631, 662, 601, 632, 633, 663, 603, 634,
        635, 665, 605, 636, 637, 667, 607, 638, 639, 669, 609, 640, 641, 670, 611,
        642, 643, 672, 613, 644, 645, 674, 615, 646, 647, 676, 617, 648, 649, 677,
        619, 650, 651, 679, 621, 652, 653, 681, 623, 654, 655, 683, 657, 684, 685,
        708, 659, 686, 687, 710, 661, 688, 689, 712, 664, 690, 691, 713, 666, 692,
        693, 715, 668, 694, 695, 717, 671, 696, 697, 718, 673, 698, 699, 720, 675,
        700, 701, 722, 678, 702, 703, 723, 680, 704, 705, 725, 682, 706, 707, 727,
        709, 728, 729, 744, 711, 730, 731, 746, 714, 732, 733, 747, 716, 734, 735,
        749, 719, 736, 737, 750, 721, 738, 739, 752, 724, 740, 741, 753, 726, 742,
        743, 755, 745, 756, 757, 764, 748, 758, 759, 765, 751, 760, 761, 766, 754,
        762, 763, 767,
    ], dtype='int')
    
    assert np.all(ref41 == nnh.dgrade(4, 1))
    assert np.all(ref42 == nnh.dgrade(4, 2))
    assert np.all(ref81 == nnh.dgrade(8, 1))
    assert np.all(ref82 == nnh.dgrade(8, 2))
    assert np.all(ref84 == nnh.dgrade(8, 4))


def test_filter9():
    ref1 = np.array([
        4, 12, 3, 2, 0, 1, 12, 5, 8, 5, 12, 0, 3, 1, 2,
        12, 6, 9, 6, 12, 1, 0, 2, 3, 12, 7, 10, 7, 12, 2,
        1, 3, 0, 12, 4, 11, 11, 7, 3, 12, 4, 0, 5, 8, 12,
        8, 4, 0, 12, 5, 1, 6, 9, 12, 9, 5, 1, 12, 6, 2,
        7, 10, 12, 10, 6, 2, 12, 7, 3, 4, 11, 12, 11, 12, 4,
        0, 8, 5, 12, 9, 10, 8, 12, 5, 1, 9, 6, 12, 10, 11,
        9, 12, 6, 2, 10, 7, 12, 11, 8, 10, 12, 7, 3, 11, 4,
        12, 8, 9,
    ], dtype='int')

    ref2 = np.array([
        4, 11, 3, 2, 0, 1, 6, 5, 13, 6, 5, 0, 3, 1, 2,
        8, 7, 15, 8, 7, 1, 0, 2, 3, 10, 9, 17, 10, 9, 2,
        1, 3, 0, 4, 11, 19, 12, 48, 11, 3, 4, 0, 5, 13, 20,
        13, 4, 0, 1, 5, 6, 48, 14, 21, 14, 48, 5, 0, 6, 1,
        7, 15, 22, 15, 6, 1, 2, 7, 8, 48, 16, 23, 16, 48, 7,
        1, 8, 2, 9, 17, 24, 17, 8, 2, 3, 9, 10, 48, 18, 25,
        18, 48, 9, 2, 10, 3, 11, 19, 26, 19, 10, 3, 0, 11, 4,
        48, 12, 27, 27, 19, 11, 48, 12, 4, 13, 20, 28, 20, 12, 4,
        0, 13, 5, 14, 21, 29, 21, 13, 5, 48, 14, 6, 15, 22, 30,
        22, 14, 6, 1, 15, 7, 16, 23, 31, 23, 15, 7, 48, 16, 8,
        17, 24, 32, 24, 16, 8, 2, 17, 9, 18, 25, 33, 25, 17, 9,
        48, 18, 10, 19, 26, 34, 26, 18, 10, 3, 19, 11, 12, 27, 35,
        28, 27, 12, 4, 20, 13, 21, 29, 36, 29, 20, 13, 5, 21, 14,
        22, 30, 37, 30, 21, 14, 6, 22, 15, 23, 31, 38, 31, 22, 15,
        7, 23, 16, 24, 32, 39, 32, 23, 16, 8, 24, 17, 25, 33, 40,
        33, 24, 17, 9, 25, 18, 26, 34, 41, 34, 25, 18, 10, 26, 19,
        27, 35, 42, 35, 26, 19, 11, 27, 12, 20, 28, 43, 43, 35, 27,
        12, 28, 20, 29, 36, 48, 36, 28, 20, 13, 29, 21, 30, 37, 44,
        37, 29, 21, 14, 30, 22, 31, 38, 48, 38, 30, 22, 15, 31, 23,
        32, 39, 45, 39, 31, 23, 16, 32, 24, 33, 40, 48, 40, 32, 24,
        17, 33, 25, 34, 41, 46, 41, 33, 25, 18, 34, 26, 35, 42, 48,
        42, 34, 26, 19, 35, 27, 28, 43, 47, 43, 48, 28, 20, 36, 29,
        37, 44, 47, 44, 36, 29, 21, 37, 30, 48, 38, 45, 37, 48, 30,
        22, 38, 31, 39, 45, 44, 45, 38, 31, 23, 39, 32, 48, 40, 46,
        39, 48, 32, 24, 40, 33, 41, 46, 45, 46, 40, 33, 25, 41, 34,
        48, 42, 47, 41, 48, 34, 26, 42, 35, 43, 47, 46, 47, 42, 35,
        27, 43, 28, 48, 36, 44, 47, 43, 36, 29, 44, 37, 38, 45, 46,
        44, 37, 38, 31, 45, 39, 40, 46, 47, 45, 39, 40, 33, 46, 41,
        42, 47, 44, 46, 41, 42, 35, 47, 43, 36, 44, 45,
    ], dtype='int')

    ref4 = np.array([
        4, 11, 3, 2, 0, 1, 6, 5, 13, 6, 5, 0, 3, 1, 2,
        8, 7, 16, 8, 7, 1, 0, 2, 3, 10, 9, 19, 10, 9, 2,
        1, 3, 0, 4, 11, 22, 12, 23, 11, 3, 4, 0, 5, 13, 25,
        13, 4, 0, 1, 5, 6, 15, 14, 26, 15, 14, 5, 0, 6, 1,
        7, 16, 29, 16, 6, 1, 2, 7, 8, 18, 17, 30, 18, 17, 7,
        1, 8, 2, 9, 19, 33, 19, 8, 2, 3, 9, 10, 21, 20, 34,
        21, 20, 9, 2, 10, 3, 11, 22, 37, 22, 10, 3, 0, 11, 4,
        12, 23, 38, 24, 39, 23, 11, 12, 4, 13, 25, 41, 25, 12, 4,
        0, 13, 5, 14, 26, 42, 26, 13, 5, 6, 14, 15, 28, 27, 43,
        28, 27, 14, 5, 15, 6, 16, 29, 45, 29, 15, 6, 1, 16, 7,
        17, 30, 46, 30, 16, 7, 8, 17, 18, 32, 31, 47, 32, 31, 17,
        7, 18, 8, 19, 33, 49, 33, 18, 8, 2, 19, 9, 20, 34, 50,
        34, 19, 9, 10, 20, 21, 36, 35, 51, 36, 35, 20, 9, 21, 10,
        22, 37, 53, 37, 21, 10, 3, 22, 11, 23, 38, 54, 38, 22, 11,
        4, 23, 12, 24, 39, 55, 40, 192, 39, 23, 24, 12, 25, 41, 56,
        41, 24, 12, 4, 25, 13, 26, 42, 57, 42, 25, 13, 5, 26, 14,
        27, 43, 58, 43, 26, 14, 15, 27, 28, 192, 44, 59, 44, 192, 27,
        14, 28, 15, 29, 45, 60, 45, 28, 15, 6, 29, 16, 30, 46, 61,
        46, 29, 16, 7, 30, 17, 31, 47, 62, 47, 30, 17, 18, 31, 32,
        192, 48, 63, 48, 192, 31, 17, 32, 18, 33, 49, 64, 49, 32, 18,
        8, 33, 19, 34, 50, 65, 50, 33, 19, 9, 34, 20, 35, 51, 66,
        51, 34, 20, 21, 35, 36, 192, 52, 67, 52, 192, 35, 20, 36, 21,
        37, 53, 68, 53, 36, 21, 10, 37, 22, 38, 54, 69, 54, 37, 22,
        11, 38, 23, 39, 55, 70, 55, 38, 23, 12, 39, 24, 192, 40, 71,
        71, 55, 39, 192, 40, 24, 41, 56, 72, 56, 40, 24, 12, 41, 25,
        42, 57, 73, 57, 41, 25, 13, 42, 26, 43, 58, 74, 58, 42, 26,
        14, 43, 27, 44, 59, 75, 59, 43, 27, 192, 44, 28, 45, 60, 76,
        60, 44, 28, 15, 45, 29, 46, 61, 77, 61, 45, 29, 16, 46, 30,
        47, 62, 78, 62, 46, 30, 17, 47, 31, 48, 63, 79, 63, 47, 31,
        192, 48, 32, 49, 64, 80, 64, 48, 32, 18, 49, 33, 50, 65, 81,
        65, 49, 33, 19, 50, 34, 51, 66, 82, 66, 50, 34, 20, 51, 35,
        52, 67, 83, 67, 51, 35, 192, 52, 36, 53, 68, 84, 68, 52, 36,
        21, 53, 37, 54, 69, 85, 69, 53, 37, 22, 54, 38, 55, 70, 86,
        70, 54, 38, 23, 55, 39, 40, 71, 87, 72, 71, 40, 24, 56, 41,
        57, 73, 88, 73, 56, 41, 25, 57, 42, 58, 74, 89, 74, 57, 42,
        26, 58, 43, 59, 75, 90, 75, 58, 43, 27, 59, 44, 60, 76, 91,
        76, 59, 44, 28, 60, 45, 61, 77, 92, 77, 60, 45, 29, 61, 46,
        62, 78, 93, 78, 61, 46, 30, 62, 47, 63, 79, 94, 79, 62, 47,
        31, 63, 48, 64, 80, 95, 80, 63, 48, 32, 64, 49, 65, 81, 96,
        81, 64, 49, 33, 65, 50, 66, 82, 97, 82, 65, 50, 34, 66, 51,
        67, 83, 98, 83, 66, 51, 35, 67, 52, 68, 84, 99, 84, 67, 52,
        36, 68, 53, 69, 85, 100, 85, 68, 53, 37, 69, 54, 70, 86, 101,
        86, 69, 54, 38, 70, 55, 71, 87, 102, 87, 70, 55, 39, 71, 40,
        56, 72, 103, 103, 87, 71, 40, 72, 56, 73, 88, 104, 88, 72, 56,
        41, 73, 57, 74, 89, 105, 89, 73, 57, 42, 74, 58, 75, 90, 106,
        90, 74, 58, 43, 75, 59, 76, 91, 107, 91, 75, 59, 44, 76, 60,
        77, 92, 108, 92, 76, 60, 45, 77, 61, 78, 93, 109, 93, 77, 61,
        46, 78, 62, 79, 94, 110, 94, 78, 62, 47, 79, 63, 80, 95, 111,
        95, 79, 63, 48, 80, 64, 81, 96, 112, 96, 80, 64, 49, 81, 65,
        82, 97, 113, 97, 81, 65, 50, 82, 66, 83, 98, 114, 98, 82, 66,
        51, 83, 67, 84, 99, 115, 99, 83, 67, 52, 84, 68, 85, 100, 116,
        100, 84, 68, 53, 85, 69, 86, 101, 117, 101, 85, 69, 54, 86, 70,
        87, 102, 118, 102, 86, 70, 55, 87, 71, 72, 103, 119, 104, 103, 72,
        56, 88, 73, 89, 105, 120, 105, 88, 73, 57, 89, 74, 90, 106, 121,
        106, 89, 74, 58, 90, 75, 91, 107, 122, 107, 90, 75, 59, 91, 76,
        92, 108, 123, 108, 91, 76, 60, 92, 77, 93, 109, 124, 109, 92, 77,
        61, 93, 78, 94, 110, 125, 110, 93, 78, 62, 94, 79, 95, 111, 126,
        111, 94, 79, 63, 95, 80, 96, 112, 127, 112, 95, 80, 64, 96, 81,
        97, 113, 128, 113, 96, 81, 65, 97, 82, 98, 114, 129, 114, 97, 82,
        66, 98, 83, 99, 115, 130, 115, 98, 83, 67, 99, 84, 100, 116, 131,
        116, 99, 84, 68, 100, 85, 101, 117, 132, 117, 100, 85, 69, 101, 86,
        102, 118, 133, 118, 101, 86, 70, 102, 87, 103, 119, 134, 119, 102, 87,
        71, 103, 72, 88, 104, 135, 135, 119, 103, 72, 104, 88, 105, 120, 136,
        120, 104, 88, 73, 105, 89, 106, 121, 137, 121, 105, 89, 74, 106, 90,
        107, 122, 138, 122, 106, 90, 75, 107, 91, 108, 123, 139, 123, 107, 91,
        76, 108, 92, 109, 124, 140, 124, 108, 92, 77, 109, 93, 110, 125, 141,
        125, 109, 93, 78, 110, 94, 111, 126, 142, 126, 110, 94, 79, 111, 95,
        112, 127, 143, 127, 111, 95, 80, 112, 96, 113, 128, 144, 128, 112, 96,
        81, 113, 97, 114, 129, 145, 129, 113, 97, 82, 114, 98, 115, 130, 146,
        130, 114, 98, 83, 115, 99, 116, 131, 147, 131, 115, 99, 84, 116, 100,
        117, 132, 148, 132, 116, 100, 85, 117, 101, 118, 133, 149, 133, 117, 101,
        86, 118, 102, 119, 134, 150, 134, 118, 102, 87, 119, 103, 104, 135, 151,
        136, 135, 104, 88, 120, 105, 121, 137, 152, 137, 120, 105, 89, 121, 106,
        122, 138, 153, 138, 121, 106, 90, 122, 107, 123, 139, 154, 139, 122, 107,
        91, 123, 108, 124, 140, 155, 140, 123, 108, 92, 124, 109, 125, 141, 156,
        141, 124, 109, 93, 125, 110, 126, 142, 157, 142, 125, 110, 94, 126, 111,
        127, 143, 158, 143, 126, 111, 95, 127, 112, 128, 144, 159, 144, 127, 112,
        96, 128, 113, 129, 145, 160, 145, 128, 113, 97, 129, 114, 130, 146, 161,
        146, 129, 114, 98, 130, 115, 131, 147, 162, 147, 130, 115, 99, 131, 116,
        132, 148, 163, 148, 131, 116, 100, 132, 117, 133, 149, 164, 149, 132, 117,
        101, 133, 118, 134, 150, 165, 150, 133, 118, 102, 134, 119, 135, 151, 166,
        151, 134, 119, 103, 135, 104, 120, 136, 167, 167, 151, 135, 104, 136, 120,
        137, 152, 192, 152, 136, 120, 105, 137, 121, 138, 153, 168, 153, 137, 121,
        106, 138, 122, 139, 154, 169, 154, 138, 122, 107, 139, 123, 140, 155, 170,
        155, 139, 123, 108, 140, 124, 141, 156, 192, 156, 140, 124, 109, 141, 125,
        142, 157, 171, 157, 141, 125, 110, 142, 126, 143, 158, 172, 158, 142, 126,
        111, 143, 127, 144, 159, 173, 159, 143, 127, 112, 144, 128, 145, 160, 192,
        160, 144, 128, 113, 145, 129, 146, 161, 174, 161, 145, 129, 114, 146, 130,
        147, 162, 175, 162, 146, 130, 115, 147, 131, 148, 163, 176, 163, 147, 131,
        116, 148, 132, 149, 164, 192, 164, 148, 132, 117, 149, 133, 150, 165, 177,
        165, 149, 133, 118, 150, 134, 151, 166, 178, 166, 150, 134, 119, 151, 135,
        136, 167, 179, 167, 192, 136, 120, 152, 137, 153, 168, 179, 168, 152, 137,
        121, 153, 138, 154, 169, 180, 169, 153, 138, 122, 154, 139, 155, 170, 181,
        170, 154, 139, 123, 155, 140, 192, 156, 171, 155, 192, 140, 124, 156, 141,
        157, 171, 170, 171, 156, 141, 125, 157, 142, 158, 172, 182, 172, 157, 142,
        126, 158, 143, 159, 173, 183, 173, 158, 143, 127, 159, 144, 192, 160, 174,
        159, 192, 144, 128, 160, 145, 161, 174, 173, 174, 160, 145, 129, 161, 146,
        162, 175, 184, 175, 161, 146, 130, 162, 147, 163, 176, 185, 176, 162, 147,
        131, 163, 148, 192, 164, 177, 163, 192, 148, 132, 164, 149, 165, 177, 176,
        177, 164, 149, 133, 165, 150, 166, 178, 186, 178, 165, 150, 134, 166, 151,
        167, 179, 187, 179, 166, 151, 135, 167, 136, 192, 152, 168, 179, 167, 152,
        137, 168, 153, 169, 180, 187, 180, 168, 153, 138, 169, 154, 170, 181, 188,
        181, 169, 154, 139, 170, 155, 156, 171, 182, 170, 155, 156, 141, 171, 157,
        172, 182, 181, 182, 171, 157, 142, 172, 158, 173, 183, 189, 183, 172, 158,
        143, 173, 159, 160, 174, 184, 173, 159, 160, 145, 174, 161, 175, 184, 183,
        184, 174, 161, 146, 175, 162, 176, 185, 190, 185, 175, 162, 147, 176, 163,
        164, 177, 186, 176, 163, 164, 149, 177, 165, 178, 186, 185, 186, 177, 165,
        150, 178, 166, 179, 187, 191, 187, 178, 166, 151, 179, 167, 152, 168, 180,
        187, 179, 168, 153, 180, 169, 181, 188, 191, 188, 180, 169, 154, 181, 170,
        171, 182, 189, 181, 170, 171, 157, 182, 172, 183, 189, 188, 189, 182, 172,
        158, 183, 173, 174, 184, 190, 183, 173, 174, 161, 184, 175, 185, 190, 189,
        190, 184, 175, 162, 185, 176, 177, 186, 191, 185, 176, 177, 165, 186, 178,
        187, 191, 190, 191, 186, 178, 166, 187, 179, 168, 180, 188, 191, 187, 180,
        169, 188, 181, 182, 189, 190, 188, 181, 182, 172, 189, 183, 184, 190, 191,
        189, 183, 184, 175, 190, 185, 186, 191, 188, 190, 185, 186, 178, 191, 187,
        180, 188, 189,
    ], dtype='int')

    assert np.all(ref1 == nnh.filter9(1))
    assert np.all(ref2 == nnh.filter9(2))
    assert np.all(ref4 == nnh.filter9(4))