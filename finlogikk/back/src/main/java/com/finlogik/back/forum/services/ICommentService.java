package com.finlogik.back.forum.services;

import com.finlogik.back.forum.entities.Comment;
import com.finlogik.back.forum.entities.CommentDTO;

import java.util.List;

public interface ICommentService {

    Comment addComment (Comment comment, String userId, Long idPost);
    Comment updateComment (Comment comment);
    void deleteComment(Long commentId);
    //List<Comment> getAllComments();
    List<CommentDTO> getAllComments();

    Comment getComment(Long commentId);

    //List<Comment> getCommentsByPostId(Long postId);
    List<CommentDTO> getCommentsByPostId(Long postId);

}
