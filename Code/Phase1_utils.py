#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

Author(s): 
Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park

Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

import numpy as np
import cv2
import dlib
from scipy import interpolate

def delaunay_triangulation(image1, image2, points1, points2, subdiv, color):
    triangles = subdiv.getTriangleList()

    triangles1= []
    triangles2 = []

    for tr in triangles:
        p1 = (int(tr[0]), int(tr[1]))
        idx1 = points1.index(p1)
        p2 = (int(tr[2]), int(tr[3]))
        idx2 = points1.index(p2)
        p3 = (int(tr[4]), int(tr[5]))
        idx3 = points1.index(p3)
        triangles1.append([int(tr[0]), int(tr[1]),int(tr[2]), int(tr[3]),int(tr[4]), int(tr[5])]) #if the points lie in the plane, then connect the dots
        triangles2.append([int(points2[idx1][0]), int(points2[idx1][1]), int(points2[idx2][0]), int(points2[idx2][1]), int(points2[idx3][0]), int(points2[idx3][1])])
        cv2.line(image1, p1, p2, color, 1, cv2.LINE_AA, 0)
        cv2.line(image1, p2, p3, color, 1, cv2.LINE_AA, 0)
        cv2.line(image1, p3, p1, color, 1, cv2.LINE_AA, 0)
        cv2.line(image2, (int(points2[idx1][0]), int(points2[idx1][1])), (int(points2[idx2][0]), int(points2[idx2][1])), color, 1, cv2.LINE_AA, 0)
        cv2.line(image2, (int(points2[idx2][0]), int(points2[idx2][1])), (int(points2[idx3][0]), int(points2[idx3][1])), color, 1, cv2.LINE_AA, 0)
        cv2.line(image2, (int(points2[idx3][0]), int(points2[idx3][1])), (int(points2[idx1][0]), int(points2[idx1][1])), color, 1, cv2.LINE_AA, 0)
    #cv2.imshow('Image 1',image1)
    #cv2.waitKey(0)
    #cv2.imshow('Image 2', image2)
    #cv2.waitKey(0)
    return triangles1, triangles2

def face_info(image1, image2 = 0):

    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #CNN detector
    detector = dlib.cnn_face_detection_model_v1('Code/Data/mmod_human_face_detector.dat')
    detected_faces = detector(image1, 1)
    rects = [detected_face.rect for detected_face in detected_faces]

    #HOG detector
    #detector = dlib.get_frontal_face_detector()
    #rects = detector(image1, 1)

    predictor = dlib.shape_predictor('Code/Data/shape_predictor_68_face_landmarks.dat')
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    points=[]

    for k, d in enumerate(rects):
        shape = predictor(gray, d)
        i = 0

        while i != 68:
            #print(len(shape.part))
            #cv2.circle(image1_copy, (shape.part(i).x, shape.part(i).y), 1, (0, 0, 255), thickness=2)
            points.append((shape.part(i).x,shape.part(i).y))
            i+=1
    #cv2.imshow('Detection',image1_copy)
    #cv2.waitKey(10)

    if image2==0:
        return points
    if len(points) == 68:
        return points,points
    points1 = []
    points2 = []
    if len(points)<136:
        return points,points
    elif len(points)>136:
        print('More than 2 faces detected')

    for i in range(0,68):
        points1.append(points[i])
    for i in range(68,136):
        points2.append(points[i])
    return points1, points2

def triangulation(image1,image2,points1,points2):
    size = image1.shape

    rect = (0, 0, size[1], size[0])
    subdiv = cv2.Subdiv2D(rect)

    for p in points1:
        subdiv.insert(p)

    copy_im1 = image1.copy()
    copy_im2 = image2.copy()

    triangles1, triangles2 = delaunay_triangulation(copy_im1, copy_im2, points1, points2, subdiv, (0, 255, 255))
    return triangles1,triangles2

def warp_using_triangulation(image1, image2, points1, points2):
    image_tmp = 0*image1
    tr1, tr2 = triangulation(image1, image2, points1, points2)

    for k in range(len(tr1)):
        c1x, c1y = tr1[k][0], tr1[k][1]
        c2x, c2y = tr1[k][2], tr1[k][3]
        c3x, c3y = tr1[k][4], tr1[k][5]
        cs1x, cs1y = tr2[k][0], tr2[k][1]
        cs2x, cs2y = tr2[k][2], tr2[k][3]
        cs3x, cs3y = tr2[k][4], tr2[k][5]

        B = np.matrix([[c1x, c2x, c3x], [c1y, c2y, c3y], [1, 1, 1]])
        A = np.matrix([[cs1x, cs2x, cs3x], [cs1y, cs2y, cs3y], [1, 1, 1]])

        triangle = np.array([[tr1[k][0], tr1[k][1]],[tr1[k][2], tr1[k][3]],[tr1[k][4], tr1[k][5]]])
        coord = get_masked(image1, triangle)

        coord_d = np.hstack((coord, np.ones((coord.shape[0], 1))))
        bayc = np.linalg.inv(B) * coord_d.T

        coord_s = A * bayc
        coord_s[0] = coord_s[0] / coord_s[2]
        coord_s[1] = coord_s[1] / coord_s[2]

        #coord_s = coord_s.astype(int)
        #coord_d = coord_d.astype(int)
        #source_indices = [coord_s[1],coord_s[0]]

        X = np.arange(0, image2.shape[1])
        Y = np.arange(0, image2.shape[0])
        b = (image2[:,:, 0])
        g = (image2[:,:, 1])
        r = (image2[:,:, 2])

        fb = interpolate.interp2d(X, Y, b, kind='cubic', fill_value=0)
        fg = interpolate.interp2d(X, Y, g, kind='cubic', fill_value=0)
        fr = interpolate.interp2d(X, Y, r, kind='cubic', fill_value=0)

        dest_indices=[]
        for i in range(0,coord.shape[0]):
            dest_indices.append([coord_d[i,1],coord_d[i,0]])

        dest_indices = np.array(dest_indices)
        dest_indices = dest_indices.astype(int)

        coord_s = coord_s.T

        # fbi = fb(coord_sx, coord_sy)
        # fgi = fg(coord_sx, coord_sy)
        # fri = fr(coord_sx, coord_sy)
        #
        # print(fbi)
        #
        # image_tmp[dest_indices[:,0],dest_indices[:,1], 0] = fbi.astype(int)
        # image_tmp[dest_indices[:,0],dest_indices[:,1], 1] = fgi.astype(int)
        # image_tmp[dest_indices[:,0],dest_indices[:,1], 2] = fri.astype(int)

        for i in range(0,coord_s.shape[0]):
            image_tmp[dest_indices[i][0],dest_indices[i][1], 0] = int(fb(coord_s[i,0],coord_s[i,1]))
            image_tmp[dest_indices[i][0],dest_indices[i][1], 1] = int(fg(coord_s[i,0],coord_s[i,1]))
            image_tmp[dest_indices[i][0],dest_indices[i][1], 2] = int(fr(coord_s[i,0],coord_s[i,1]))

    return image_tmp


def get_masked(img, kpt):
    tolerance = 1e-12
    from scipy.spatial import ConvexHull
    hull = ConvexHull(kpt)
    h, w, _ = img.shape
    Y, X = np.indices((h, w))
    xmin = np.int(np.min(kpt[:, 0]))
    xmax = np.int(np.max(kpt[:, 0])) + 1
    ymin = np.int(np.min(kpt[:, 1]))
    ymax = np.int(np.max(kpt[:, 1])) + 1
    X = X[ymin:ymax, xmin:xmax]
    Y = Y[ymin:ymax, xmin:xmax]
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    P = np.hstack((X, Y))
    in_hull = np.all(np.add(np.dot(P, hull.equations[:, :-1].T), hull.equations[:, -1]) <= tolerance, axis=1)
    coords = P[in_hull]

    return coords

def check_params(kpt1,kpt2,wax,way):
    # Obtaining K
    dX = kpt1[:,0].reshape(1,-1) - kpt1[:,0].reshape(-1,1)
    dY = kpt1[:,1].reshape(1,-1) - kpt1[:,1].reshape(-1,1)
    r2 = dX*dX + dY*dY
    K = r2 * np.log(r2)
    K[r2 == 0] = 0
    # Obtaining P
    P = np.hstack((kpt1,np.ones((kpt1.shape[0],1))))

    # Decalring combined matrix
    M = np.hstack((K,P))

    # Obtaining points2
    x = M @ wax
    y = M @ way
    res = kpt2 - np.hstack((x.reshape(-1,1),y.reshape(-1,1)))
    print(np.linalg.norm(res))


def get_TPS_params(kpt1, kpt2):
    """
    Will give map parameters to
    transform kpt1 to kpt2
    i.e., kpt2 = f_TPS(kpt1)
    """
    p = kpt1.shape[0]

    # Obtaining K
    dX = kpt1[:, 0].reshape(-1, 1) - kpt1[:, 0].reshape(1, -1)
    dY = kpt1[:, 1].reshape(-1, 1) - kpt1[:, 1].reshape(1, -1)
    r2 = dX * dX + dY * dY
    K = r2 * np.log(r2)
    K[r2 == 0] = 0

    # Obtaining P
    P = np.hstack((kpt1, np.ones((kpt1.shape[0], 1))))

    # Obtaining V
    Vx = np.vstack((kpt2[:, 0].reshape(-1, 1), np.zeros((3, 1))))
    Vy = np.vstack((kpt2[:, 1].reshape(-1, 1), np.zeros((3, 1))))

    # Declaring combined matrix
    M = np.zeros((p + 3, p + 3))
    M[0:p, 0:p] = K
    M[0:p, p:] = P
    M[p:, 0:p] = P.T
    Lambda = 1e-3
    M += Lambda * np.identity(p + 3)

    # Solving for parameters
    wax = np.linalg.inv(M) @ Vx
    way = np.linalg.inv(M) @ Vy
    return wax, way

def iwarp(img1,img2,kpt1,points1,wax,way):
    """
    replaced face in img1
    with face in img2
    """
    # Obtaining K
    dX = kpt1[:,0].reshape(1,-1) - points1[:,0].reshape(-1,1)
    dY = kpt1[:,1].reshape(1,-1) - points1[:,1].reshape(-1,1)
    r2 = dX*dX + dY*dY
    K = r2 * np.log(r2)
    K[r2 == 0] = 0

    # Obtaining P
    P = np.hstack((points1,np.ones((points1.shape[0],1))))

    # Decalring combined matrix
    M = np.hstack((K,P))

    # Obtaining points2
    x = M @ wax
    y = M @ way
    x = x.reshape(-1,)
    y = y.reshape(-1,)
    x = x.astype(int)
    y = y.astype(int)
    x[x < 0] = 0
    y[y < 0] = 0
    x[x >= img2.shape[1]] = -1
    y[y >= img2.shape[0]] = -1
    # Replacing values at points1 in img1
    # to values at points2 in img2
    warped_img = np.zeros(img1.shape,dtype=np.uint8) #img1.copy()
    points1 = points1.astype(int)
    warped_img[points1[:,1],points1[:,0]] = img2[y,x]
    return warped_img

def blending( image1, points1, warped_img):
    face_mask = np.zeros(image1.shape, image1.dtype)
    convexhull2 = cv2.convexHull(points1)
    (x, y, width, height) = cv2.boundingRect(convexhull2)
    center = (int((x + x + width) / 2), int((y + y + height) / 2))
    mask = cv2.fillConvexPoly(face_mask, convexhull2, [255, 255, 255])
    normal_clone = cv2.seamlessClone(warped_img, image1, mask, center, cv2.NORMAL_CLONE)
    return normal_clone

def swap_with_ref_image(Test_video_path, image_path, output_path,warping_technique):

    KF = None

    vid_capture = cv2.VideoCapture(Test_video_path)
    image2 = cv2.imread(image_path)

    fps = vid_capture.get(cv2.CAP_PROP_FPS)
    dt = 1/fps

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (854, 480))

    #Kalman Filter matrix initialization
    P = np.zeros((68, 4, 4))
    x_vec = np.zeros((68, 4))

    H = np.array([[1,0,0,0],[0,1,0,0]])
    F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    for i in range(0,68):
        P[i] = np.array([[10, 0, 0, 0], [0, 10, 0, 0], [0, 0, 10, 0], [0, 0, 0, 10]])
    Q = np.array([[0.1, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0.1]])
    R = np.array([[1, 0], [0, 1]])
    I = np.eye(4)

    k=0

    while (vid_capture.isOpened()):
        ret, frame = vid_capture.read()

        if ret == True:
            image1 = frame

            points1 = face_info(image1)
            points2 = face_info(image2)

            points_1=[]

            if len(points1)<68:
                out.write(image1)
                cv2.imshow('Output Frame', image1)
                cv2.waitKey(10)
                print('Frame skipped')
                continue

            if k == 0:
                for i in range(0,68):
                    x = np.array([[points1[i][0]],[points1[i][1]],[0],[0]])
                    x_vec[i][0] = x[0]
                    x_vec[i][1] = x[1]
                    x_vec[i][2] = x[2]
                    x_vec[i][3] = x[3]
            else:
                for i in range(0,68):
                    x = np.array([x_vec[i][0], x_vec[i][1], x_vec[i][2], x_vec[i][3]])
                    z = np.array([points1[i][0],points1[i][1]])
                    x_ = np.matmul(F, x.T)
                    P_ = F * P[i] * F.T + Q
                    y = z - np.matmul(H, x_)
                    S = np.matmul(np.matmul(H , P_), H.T) + R
                    K = np.matmul(np.matmul(P_, H.T), np.linalg.inv(S))
                    x = x_ + np.matmul(K, y)
                    P[i] = np.matmul((I - np.matmul(K, H)), P_)
                    points_1.append((int(x[0]),int(x[1])))
                    x_vec[i][0] = x[0]
                    x_vec[i][1] = x[1]
                    x_vec[i][2] = x[2]
                    x_vec[i][3] = x[3]

            if k == 0:
                points_1 = points1
            if KF == None:
                points_1=points1

            if warping_technique == 'Triangulation':
                warped_img_triangulation = warp_using_triangulation(image1, image2, points_1, points2)
                points_1 = np.array(points_1)
                blend_img_triangulation = blending(image1, points_1, warped_img_triangulation)

                out.write(blend_img_triangulation)
                cv2.imshow('Output Frame', blend_img_triangulation)
                cv2.waitKey(0)
            else:
                points_1 = np.array(points_1)
                points2 = np.array(points2)
                wax, way = get_TPS_params(points_1, points2)
                #check_params(points_1,points2,wax, way)
                coords = get_masked(image1, points_1)
                warped_img_TPS = iwarp(image1, image2, points_1, coords, wax, way)
                blend_img_TPS = blending(image1, points_1, warped_img_TPS)

                out.write(blend_img_TPS)
                cv2.imshow('Output Frame', blend_img_TPS)
                cv2.waitKey(0)
            k+=1
        else:
            break

def swap_within_video(Test_video_path, output_path,warping_technique):

    vid_capture = cv2.VideoCapture(Test_video_path)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))
    k=0

    while (vid_capture.isOpened()):
        ret, frame = vid_capture.read()

        if ret == True and k<=50:
            image1 = frame
            image1_copy = image1.copy()
            print(image1.shape)
            points_1, points2 = face_info(image1, 1)

            if len(points_1)!=68:
                out.write(image1)
                cv2.imshow('Output Frame', image1)
                cv2.waitKey(10)
                print('Frame skipped')
                k=k+1
                continue

            if warping_technique == 'Triangulation':
                warped_img_triangulation = warp_using_triangulation(image1, image1_copy, points_1, points2)

                points_1 = np.array(points_1)
                blend_img_triangulation = blending(image1, points_1, warped_img_triangulation)
                warped_img_triangulation = warp_using_triangulation(warped_img_triangulation, image1_copy, points2, points_1)
                points2 = np.array(points2)
                blend_img_triangulation = blending(blend_img_triangulation, points2, warped_img_triangulation)

                out.write(blend_img_triangulation)
                cv2.imshow('Output Frame', blend_img_triangulation)
                cv2.waitKey(10)

            else:
                points_1 = np.array(points_1)
                points2 = np.array(points2)

                wax, way = get_TPS_params(points_1, points2)
                coords = get_masked(image1, points_1)
                warped_img_TPS = iwarp(image1, image1_copy, points_1, coords, wax, way)
                blend_img_TPS = blending(image1, points_1, warped_img_TPS)
                wax, way = get_TPS_params(points2, points_1)
                coords = get_masked(image1_copy, points2)
                warped_img_TPS = iwarp(warped_img_TPS, image1_copy, points2, coords, wax, way)
                blend_img_TPS = blending(blend_img_TPS, points2, warped_img_TPS)

                out.write(blend_img_TPS)
                cv2.imshow('Output Frame', blend_img_TPS)
                cv2.waitKey(10)
            k+=1

        else:
            break

 
