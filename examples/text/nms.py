import numpy as np
import os
import shapely
from shapely.geometry import Polygon,MultiPoint

def polygon_from_list(line):
    """
    Create a shapely polygon object from gt or dt line.
    """
    #polygon_points = [float(o) for o in line.split(',')[:8]]
    polygon_points = np.array(line).reshape(4, 2)
    polygon = Polygon(polygon_points).convex_hull
    return polygon

def list_from_str(st):
    line = st.split(',')
    new_line = [float(a) for a in line[0:8]]+[float(line[-1])]
    return new_line

def polygon_iou(list1, list2):
    """
    Intersection over union between two shapely polygons.
    """
    polygon_points1 = np.array(list1).reshape(4, 2)
    poly1 = Polygon(polygon_points1).convex_hull
    polygon_points2 = np.array(list2).reshape(4, 2)
    poly2 = Polygon(polygon_points2).convex_hull
    union_poly = np.concatenate((polygon_points1,polygon_points2))
    if not poly1.intersects(poly2): # this test is fast and can accelerate calculation
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            #union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            if union_area == 0:
                return 0
            iou = float(inter_area) / union_area
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou

def nms(boxes,overlap):
    rec_scores = [b[-1] for b in boxes]
    indices = sorted(range(len(rec_scores)), key=lambda k: -rec_scores[k])
    box_num = len(boxes)
    nms_flag = [True]*box_num
    for i in range(box_num):
        ii = indices[i]
        if not nms_flag[ii]:
            continue
        for j in range(box_num):
            jj = indices[j]
            if j == i:
                continue
            if not nms_flag[jj]:
                continue
            box1 = boxes[ii]
            box2 = boxes[jj] 
            box1_score = rec_scores[ii]
            box2_score = rec_scores[jj] 
            # str1 = box1[9] 
            # str2 = box2[9]
            box_i = [box1[0],box1[1],box1[4],box1[5]]
            box_j = [box2[0],box2[1],box2[4],box2[5]]
            poly1 = polygon_from_list(box1[0:8])
            poly2 = polygon_from_list(box2[0:8])
            iou = polygon_iou(box1[0:8],box2[0:8])
            thresh = overlap
     
            if iou > thresh:
                if box1_score > box2_score:
                    nms_flag[jj] = False  
                if box1_score == box2_score and poly1.area > poly2.area:
                    nms_flag[jj] = False  
                if box1_score == box2_score and poly1.area<=poly2.area:
                    nms_flag[ii] = False  
                    break
            '''
            if abs((box_i[3]-box_i[1])-(box_j[3]-box_j[1]))<((box_i[3]-box_i[1])+(box_j[3]-box_j[1]))/2:
                if abs(box_i[3]-box_j[3])+abs(box_i[1]-box_j[1])<(max(box_i[3],box_j[3])-min(box_i[1],box_j[1]))/3:
                    if box_i[0]<=box_j[0] and (box_i[2]+min(box_i[3]-box_i[1],box_j[3]-box_j[1])>=box_j[2]):
                        nms_flag[jj] = False
            '''
    return nms_flag

def test_single(dt_dir,score,overlap,save_dir):
    for i in range(1,233):
        print(i)
        with open(os.path.join(dt_dir,'res_img_'+str(i)+'.txt'),'r') as f:
          dt_lines = [a.strip() for a in f.readlines()]
        dt_lines = [list_from_str(dt) for dt in dt_lines]

        dt_lines = [dt for dt in dt_lines if dt[8]>score]
        dt_lines = sorted(dt_lines, key=lambda x:-float(x[8]))
        nms_flag = nms(dt_lines,overlap)
        boxes = []
        for k,dt in enumerate(dt_lines):
            if nms_flag[k]:
                if dt[8] > score:
                    if dt not in boxes:
                        boxes.append(dt)

        with open(os.path.join(save_dir,'res_img_'+str(i)+'.txt'),'w') as f:
            for box in boxes:
                box = [int(b) for b in box]
                f.write(str(box[0])+','+str(box[1])+','+str(box[4])+','+str(box[5])+'\r\n')
                # f.write(str(box[0])+','+str(box[1])+','+str(box[2])+','+str(box[3])+','+str(box[4])+','+str(box[5])+','+str(box[6])+','+str(box[7])+'\r\n')

if __name__ == '__main__':
    dt_dir = '/home/mhliao/research/oriented/TextBoxes_polygon/data/data_text/test_bb_ic13_ms/'
    score_det = 0.3
    overlap = 0.2
    save_dir = '/home/mhliao/research/oriented/TextBoxes_polygon/data/data_text/test_bb_ic13_submit'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    test_single(dt_dir,score_det,overlap,save_dir)